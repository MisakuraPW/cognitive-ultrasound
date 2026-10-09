"""Frozen one-episode audit; never applies updates to a production checkpoint."""
import argparse
import copy
import inspect
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from cognitive_ultrasound.preparation.common import atomic_json, read_json
from cognitive_ultrasound.task_budget.data import read_episode
from cognitive_ultrasound.task_budget.episode import rollout, gs_local_objective
from cognitive_ultrasound.task_budget.experiment import setup
from cognitive_ultrasound.task_budget.perception import exact_forward_vjp
from cognitive_ultrasound.task_budget.policy import restore, st_weights, rl_loss, probabilities, adam, adam_state
from cognitive_ultrasound.task_budget.suite import process_alive


def norms(tree):
    return {h:float(jnp.sqrt(sum(jnp.sum(v*v) for v in t.values()))) for h,t in tree.items()}


def second_stage_scalar(params, context, adjoint, sampler, cfg, action, length):
    # First budget/history/ranking fixed; second prefix excludes first-stage acquired lines.
    c=dict(context,action1=action,k2=cfg["budgets"]["second"][action])
    value,_=gs_local_objective(params,c,adjoint,sampler,cfg,1.0,2.0,length)
    return float(value)


class DetachedSampler:
    def __init__(self, parent):self.parent=parent
    def infer_for_replay(self,h,m,p,key):
        return jax.lax.stop_gradient(self.parent.infer(jax.lax.stop_gradient(h),jax.lax.stop_gradient(m),jax.lax.stop_gradient(p),key))
    infer=infer_for_replay


class ZeroGuidanceSampler:
    def __init__(self,parent):
        from ulsa.agent import setup_agent
        fn=getattr(parent.posterior,"__wrapped__",None) or getattr(parent.posterior,"_fun")
        closed=inspect.getclosurevars(fn).nonlocals
        config=copy.deepcopy(closed["agent_config"])
        config.diffusion_inference.guidance_kwargs["omega"]=0.0
        agent,_=setup_agent(config,jax.random.PRNGKey(0),model=closed["model"],jit_mode="posterior_sample")
        self.kernel=jax.jit(jax.checkpoint(agent.recover.keywords["posterior_sample"]))
        self.replay=exact_forward_vjp(self.kernel)
    def infer_for_replay(self,h,m,p,key):return self.replay(jnp.asarray(h),jnp.asarray(m),jnp.asarray(p),key)
    infer=infer_for_replay


def rl_direction_test(params,cfg,context):
    c=dict(context,cold=False)
    records=[]
    for advantage in (1.,-1.):
        grad=jax.grad(rl_loss)(params,[c],advantage)
        changed,_,_=adam(params,grad,adam_state(params),cfg)
        for index,head in enumerate(("first","second")):
            action=c[f"action{index}"]
            before=float(probabilities(params[head],c[f"state{index}"],c[f"legal{index}"])[action])
            after=float(probabilities(changed[head],c[f"state{index}"],c[f"legal{index}"])[action])
            records.append(dict(head=head,advantage=advantage,before=before,after=after,
                                passed=after>before if advantage>0 else after<before))
    return records


def run(args):
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    state=read_json(source/"status.json")
    if process_alive(state.get("pid"),state.get("created")) or process_alive(state.get("worker_pid"),state.get("worker_created")):
        raise RuntimeError("Pause production before the isolated GPU audit")
    cfg=read_json(source/"config.json");manifest=read_json(source/"manifest.json")
    job=source/"jobs/E1_l1_s42_train"
    record=read_json(job/"updates/00002.json")
    params,_,_=restore(job/"updates/00001.npz")
    temperature=cfg["training"]["temperature_start"]*(cfg["training"]["temperature_end"]/cfg["training"]["temperature_start"])**(1/(cfg["training"]["updates"]-1))
    perception,task=setup(cfg,out)
    result=dict(kind="ISOLATED_LEARNING_DIAGNOSTIC_NOT_A_NEW_BASELINE",episode_update=2,frames=[],
                production_states_advanced=False,omega0_scope="numerical-path ablation ONLY; not an adoption or quality comparison")
    try:
        images,rows,contexts=rollout(cfg,perception,task,
            read_episode(cfg,manifest,record["case"],record["start"],record["frames"]),
            params,"E1",record["action_seed"],training=True)
        prediction,adjoint,_=task.video(images,gradient=True)
        adjoint=adjoint*np.sign(prediction-record["ef_truth"])
        result["replay"]=dict(prediction=prediction,recorded_prediction=record["ef_prediction"],
                            mean_lines=float(np.mean([r["k1"]+r["k2"] for r in rows])),
                            passed=abs(prediction-record["ef_prediction"])<2e-4)
        if not result["replay"]["passed"]:raise RuntimeError("Recorded episode does not replay")
        result["rl_sign_test"]=rl_direction_test(params,cfg,contexts[16])
        atomic_json(out/"result.partial.json",result)
        zero=ZeroGuidanceSampler(perception)
        for index in (16,32):
            c=contexts[index];g=jnp.asarray(adjoint[index]);frame=dict(index=index,budgets=[c["k1"],c["k2"]],modes={})
            for name,sampler in (("original",perception),("detached_sampler",DetachedSampler(perception)),("omega0_diagnostic",zero)):
                f=lambda p:gs_local_objective(p,c,g,sampler,cfg,temperature,0.,len(contexts))
                direct,expected=f(params)
                start=time.perf_counter();(value,actual),gradient=jax.value_and_grad(f,has_aux=True)(params)
                delta=float(np.max(np.abs(np.asarray(expected)-np.asarray(actual))))
                frame["modes"][name]=dict(task_gradient_norms=norms(gradient),ad_forward_delta=delta,
                    scalar_value_delta=float(value-direct),seconds=time.perf_counter()-start,
                    finite=bool(all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(gradient))))
                print(json.dumps(dict(event="FRAME_PATH",frame=index,mode=name,**frame["modes"][name])),flush=True)
            # Sampling-map directional derivative, with a fixed binary mask/previous state/key.
            m0=jnp.broadcast_to(jnp.asarray(c["bank0"])[c["action0"]],(112,112))
            h=jnp.concatenate([jnp.asarray(c["history"])[...,1:],(m0*jnp.asarray(c["target"]))[...,None]],axis=-1)
            m=jnp.concatenate([jnp.asarray(c["masks"])[...,1:],m0[...,None]],axis=-1)
            direction=np.zeros(tuple(h.shape),np.float32)
            direction[...,-1]=np.asarray(m0)*np.random.default_rng(123).choice([-1.,1.],size=(112,112))
            direction=jnp.asarray(direction)/jnp.linalg.norm(jnp.asarray(direction))
            kernel=perception.warm
            def scalar(history):
                sample=kernel(history,m,jnp.asarray(c["previous"]),c["key0"])
                image=sample[0,...,-1]
                return jnp.sum(image*g),image
            base,image=scalar(h);(ad_value,ad_image),dh=jax.value_and_grad(scalar,has_aux=True)(h)
            derivative=float(jnp.sum(dh*direction))
            fd=[]
            for eps in (1e-2,1e-3,1e-4):
                plus,_=scalar(h+eps*direction);minus,_=scalar(h-eps*direction)
                fd.append(dict(epsilon=eps,directional_derivative=float((plus-minus)/(2*eps))))
            frame["sampling_map_derivative"]=dict(ad_directional=derivative,finite_difference=fd,
                 ad_primal_delta=float(np.max(np.abs(np.asarray(ad_image)-np.asarray(image)))),
                 scope="fixed binary mask, measurement perturbation; not finite differences of hard categorical actions")
            values=[second_stage_scalar(params,c,g,perception,cfg,a,len(contexts)) for a in range(len(cfg["budgets"]["second"]))]
            frame["second_budget_counterfactual"]=dict(budgets=cfg["budgets"]["second"],local_linear_task_plus_cost=values,
                 chosen=c["k2"],scope="fixed first budget/history/noise/ranking and EF adjoint; not full-video EF reassessment")
            result["frames"].append(frame);atomic_json(out/"result.partial.json",result)
        result["status"]="completed";atomic_json(out/"result.json",result)
    finally:task.close()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--source",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
