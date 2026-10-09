"""CPU synthetic checks of the declared repair, not scientific/GPU effectiveness evidence."""
import copy
from collections import Counter
import importlib.util
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cognitive_ultrasound.config import ROOT,load
from cognitive_ultrasound.task_budget.episode import rollout,gs_gradient,projection_objective
from cognitive_ultrasound.task_budget.policy import initialize,logits,adam,adam_state,rl_loss,restore
from cognitive_ultrasound.task_budget.protocol import state_features
from cognitive_ultrasound.task_budget.repair import joint_budget_schedule,repair_jobs
from cognitive_ultrasound.task_budget import experiment


@pytest.fixture
def cfg():
    c=load(ROOT/"configs/task_budget_ef_repair.yaml")
    c["feature_scales"]=[.1]*12
    return c


class Model:
    def __init__(self):self.calls=0
    def infer(self,h,m,p,key,cold=False):
        self.calls+=1;h=jnp.asarray(h)
        return jnp.stack([.5*h-.1,.5*h+.1])


class Task:
    def __init__(self):self.closed=False;self.calls=0;self.seconds=0.
    def score(self,p,h):return np.linspace(.001,.002,112),np.array([50.,51.])
    def video(self,images,gradient=False,domain="polar"):
        self.calls+=1
        value=float(50+np.mean(images));g=np.ones_like(images)/images.size
        return value,g,[value]
    def close(self):self.closed=True


def frames(n=4):return np.broadcast_to(np.linspace(-.8,.8,112),(n,112,112)).copy().astype(np.float32)


@pytest.mark.parametrize("second",[0,2,14])
def test_projection_replays_actual_images_and_never_redifferentiates_sampler(cfg,second):
    p=initialize(42,cfg);m=Model();task=Task()
    images,_,contexts=rollout(cfg,m,task,frames(),p,"E0",42,fixed=(4,second))
    before=m.calls;_,g,_=task.video(images,True)
    derivative,gate=gs_gradient(p,contexts,g,m,cfg,.7,2.)
    assert m.calls==before and gate["hard_replay_max_abs"]==0
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(derivative))
    assert all(x>0 for x in gate["task_gradient_norms"].values())


def test_projection_derivative_matches_its_declared_continuous_surrogate(cfg):
    p=initialize(42,cfg);images,_,contexts=rollout(cfg,Model(),Task(),frames(),p,"E0",42,fixed=(4,2))
    c=contexts[1];g=jnp.ones((112,112))/.5e5
    a=jax.grad(lambda p:projection_objective(p,c,g,cfg,.8,2.,4)[0])(p)
    def relaxed(q):
        w=[]
        for i,h in enumerate(("first","second")):
            w.append(jax.nn.softmax((logits(q[h],c[f"state{i}"])+c[f"noise{i}"])/.8))
        m0=w[0]@c["bank0"];m1=w[1]@c["bank1"]
        r0=jnp.asarray(c["bank0"])[c["action0"]];r1=jnp.asarray(c["bank1"])[c["action1"]]
        delta=jnp.broadcast_to((1-r1)*(m0-r0)+(1-r0)*(m1-r1),(112,112))
        innovation=c["target"]-np.clip(c["raw_prediction"],-1,1)
        image=c["prediction"]+delta*innovation
        return jnp.sum(image*g)+2*(w[0]@jnp.asarray(cfg["budgets"]["first"])+w[1]@jnp.asarray(cfg["budgets"]["second"]))/(112*4)
    b=jax.grad(relaxed)(p)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x,y,atol=2e-6,rtol=2e-5)
    direction=jnp.array([1.,-1.,0.,0.]);eps=.001
    plus=copy.deepcopy(p);minus=copy.deepcopy(p)
    plus["first"]["b2"]=plus["first"]["b2"]+eps*direction
    minus["first"]["b2"]=minus["first"]["b2"]-eps*direction
    fd=float((relaxed(plus)-relaxed(minus))/(2*eps))
    np.testing.assert_allclose(fd,float(jnp.sum(a["first"]["b2"]*direction)),atol=2e-5,rtol=.01)


def test_training_feature_scales_lift_small_inputs_and_reject_bad_scales():
    particles=np.zeros((2,112,112),np.float32);scores=np.ones(112)*1e-6
    args=(particles,np.zeros((112,112)),scores,np.array([50.,50.]))
    old=state_features(*args);s=np.ones(12);s[8:10]=1e-6
    new=state_features(*args,feature_scales=s)
    assert new[8]>.6 and old[8]<2e-6
    with pytest.raises(ValueError):state_features(*args,feature_scales=[0]*12)


@pytest.mark.parametrize("pairs",[[(4,2)]*8,[(4,0),(14,14),(4,2),(14,14),(4,0),(7,4)]])
def test_control_matches_joint_counts_and_phase_costs(pairs):
    rows=[dict(k1=10,k2=4)]+[dict(k1=x,k2=y) for x,y in pairs]
    schedule=joint_budget_schedule(rows)
    assert tuple(schedule[0])==(10,4)
    assert Counter(map(tuple,schedule[1:]))==Counter(pairs)
    assert (schedule.sum(0)==np.array([(r['k1'],r['k2']) for r in rows]).sum(0)).all()
    assert sum(schedule[:,1]>0)==sum(r['k2']>0 for r in rows)
    if len(set(pairs))==1:assert list(map(tuple,schedule[1:]))==pairs


def test_repair_schedule_freezes_one_point_and_continues_milestones(cfg):
    plan=repair_jobs(cfg);train=[s for s in plan if s['kind']=='train']
    assert len(train)==6
    for h in ['E1','E2']:
        tasks=[s for s in train if s['method']==h]
        assert [s['end_update'] for s in tasks]==[50,100,200]
        assert tasks[1]['resume_parent']==tasks[0]['id'] and tasks[2]['resume_parent']==tasks[1]['id']
    confirmation=[i for i,s in enumerate(plan) if s.get('cohort')=='confirmation']
    assert min(confirmation)>max(i for i,s in enumerate(plan) if s.get('cohort')=='development')


def test_rl_constant_scaling_preserves_gradient_direction(cfg):
    p=initialize(42,cfg);_,_,contexts=rollout(cfg,Model(),Task(),frames(),p,'E2',42,training=True)
    a=jax.grad(rl_loss)(p,contexts,2.,1.)
    b=jax.grad(rl_loss)(p,contexts,2.,126.)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):
        np.testing.assert_allclose(x/126,y,atol=1e-7,rtol=1e-5)


def test_action_independent_reference_preserves_expected_policy_gradient():
    logits=jnp.array([.1,.2,-.1,.3]);rewards=jnp.array([-2.,-4.,-1.,-3.])
    def gradient(reference):
        p=jax.nn.softmax(logits)
        return sum(p[i]*jax.grad(lambda z:-jax.nn.log_softmax(z)[i]*(rewards[i]+reference))(logits) for i in range(4))
    np.testing.assert_allclose(gradient(0.),gradient(3.),rtol=1e-5,atol=1e-6)


def test_milestone_resume_restores_optimizer_rng_and_excess_reward_baseline(cfg,tmp_path,monkeypatch):
    cfg=copy.deepcopy(cfg);cfg['training']['updates']=4;cfg['training']['temperature_updates']=4
    cfg['training']['clip_frames']=64
    class LongerModel(Model):pass
    monkeypatch.setattr(experiment,'setup',lambda c,o:(LongerModel(),Task()))
    monkeypatch.setattr(experiment,'read_episode',lambda *args:frames(64))
    manifest=dict(cohorts=dict(train=['case']),files=dict(case=dict(frames=64,ef=50.)))
    spec=dict(method='E2',seed=42,**{'lambda':2.})
    for n in ['continuous','first','second']:(tmp_path/n).mkdir()
    experiment.train(spec,cfg,manifest,tmp_path,tmp_path/'continuous')
    experiment.train(spec,cfg,manifest,tmp_path,tmp_path/'first',end_update=2)
    experiment.train(dict(spec,resume_checkpoint=str(tmp_path/'first/policy.npz')),cfg,manifest,tmp_path,tmp_path/'second',end_update=4)
    for x,y in zip(jax.tree_util.tree_leaves(restore(tmp_path/'continuous/policy.npz')),jax.tree_util.tree_leaves(restore(tmp_path/'second/policy.npz'))):
        np.testing.assert_array_equal(x,y)
    assert sorted(p.stem for p in (tmp_path/'second/updates').glob('*.npz'))==['00003','00004']


def test_feature_calibration_rejects_development_patient(cfg,tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('prepare_repair',ROOT/'scripts/prepare_task_budget_repair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'old';source.mkdir();(source/'status.json').write_text(json.dumps(dict(status='stopped')))
    (source/'config.json').write_text('{}')
    engineering=tmp_path/'engineering';d=engineering/'final_controls/E1_original'
    (d/'audit').mkdir(parents=True);(d/'updates').mkdir()
    np.savez(d/'audit/00001.npz',state_before=np.ones((3,12)),state_intermediate=np.ones((3,12)))
    (d/'updates/00001.json').write_text(json.dumps(dict(case='validation_patient',update=1)))
    monkeypatch.setattr(module,'configuration',lambda p:copy.deepcopy(cfg))
    monkeypatch.setattr(module,'lock_manifest',lambda c,r:dict(cohorts=dict(train=['training_patient'])))
    with pytest.raises(ValueError,match='non-TRAIN'):
        module.prepare(source,engineering,tmp_path/'new','unused')
    assert not (tmp_path/'new/config.json').exists()


def test_report_keeps_training_checkpoints_separate(cfg,tmp_path,monkeypatch):
    from cognitive_ultrasound.task_budget import report as module
    from cognitive_ultrasound.preparation.common import atomic_json
    cfg=copy.deepcopy(cfg);cfg['functional_fixture']=True
    atomic_json(tmp_path/'config.json',cfg)
    names=[f'case{i}' for i in range(8)]
    atomic_json(tmp_path/'manifest.json',dict(cohorts=dict(development=names,confirmation=['testcase'])))
    jobs=[]
    for n in (50,100):
        j=dict(id=f'E1_{n}',kind='evaluate',method='E1',cohort='development',seed=42,checkpoint_update=n,case_limit=4,**{'lambda':2.})
        jobs.append(j)
        records=[]
        for case in names[:4]:
            r=dict(case=case,absolute_error=n/100,mean_lines=10.,seconds=1.,mean_psnr=20.,mean_ssim=.8,mean_mae=.1,prediction_preservation_error=1.,perception_calls=2.,full_input_absolute_error=.2)
            records.append(r)
        atomic_json(tmp_path/'jobs'/j['id']/'result.json',dict(status='completed',records=records))
    atomic_json(tmp_path/'plan.json',jobs)
    monkeypatch.setattr(module,'figures',lambda *a,**k:None)
    points=module.report(tmp_path)
    assert len(points)==2 and all(p['cases']==4 and p['complete'] for p in points)
    assert {p['absolute_error'] for p in points}=={.5,1.}
