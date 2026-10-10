"""Read-only analysis of the downloaded EF budget repair batch; no GPU work."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(values):
    values = np.asarray(values, dtype=float)
    sample = np.random.default_rng(20261008).integers(len(values), size=(5000, len(values)))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[sample].mean(1), [.025, .975]).tolist())


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results/task_budget_ef_repair_v4_download/task_budget_ef_repair_v4"))
    parser.add_argument("--output", type=Path, default=Path("reports/task_budget_ef_repair_20261010"))
    args = parser.parse_args()
    r, out = args.source.resolve(), args.output.resolve()
    assert r != out and not out.is_relative_to(r)
    out.mkdir(parents=True, exist_ok=True)
    jobs = r / "jobs"
    status, summary, manifest = [read(r / name) for name in ["status.json", "summary.json", "manifest.json"]]
    assert status["status"] == "completed" and not status["failures"]
    assert len(summary["ledger"]) == 18 and all(x["status"] == "completed" for x in summary["ledger"])
    cohorts = manifest["cohorts"]
    assert all(not (set(cohorts[a]) & set(cohorts[b])) for a,b in [("train","development"),("train","confirmation"),("development","confirmation")])
    source_hashes = {}

    def loaded(p):
        source_hashes[str(p.relative_to(r)).replace("\\", "/")] = digest(p)
        return read(p)

    def records(name):
        return loaded(jobs / name / "result.json")["records"]

    confirm = {m: records(f"{m}_confirmation" if m != "E0" else "E0_10_4_s42_confirmation") for m in ["E0", "E1", "E2"]}
    case_ids = [x["case"] for x in confirm["E0"]]
    assert len(case_ids) == 16 and set(case_ids) == set(cohorts["confirmation"])
    cases = []
    comparison = {}
    timing = {}
    trajectories = {}
    for method, rows in confirm.items():
        assert [x["case"] for x in rows] == case_ids
        job = f"{method}_confirmation" if method != "E0" else "E0_10_4_s42_confirmation"
        hist, switches, frame_times, cold_times = Counter(), [], [], []
        source_cases = {}
        for row in rows:
            assert row["seed"] == 42
            d = jobs / job / Path(row["case"]).stem
            sequence = loaded(d / "trajectory.json")
            assert len(sequence) == row["frames"]
            assert (sequence[0]["k1"], sequence[0]["k2"]) == (10,4)
            assert sum(x["actual_lines"] for x in sequence) == row["total_lines"]
            for x in sequence:
                assert len(x["lines1"]) == x["k1"] and len(x["lines2"]) == x["k2"]
                assert len(set(x["lines1"] + x["lines2"])) == x["actual_lines"]
                assert 0 <= min(x["lines1"] + x["lines2"]) and max(x["lines1"] + x["lines2"]) < 112
                assert x["perception_calls"] == 1 + int(x["k2"] > 0)
            hist.update((x["k1"],x["k2"]) for x in sequence[1:])
            switches.append(row["budget_switch_rate"])
            frame_times.extend(x["frame_wall_s"] for x in sequence[1:])
            cold_times.append(sequence[0]["frame_wall_s"])
            source_cases[row["case"]] = sequence
            if method != "E0":
                control = loaded(d / "matched_trajectory.json")
                assert len(control) == len(sequence)
                assert Counter((x["k1"],x["k2"]) for x in control[1:]) == Counter((x["k1"],x["k2"]) for x in sequence[1:])
                assert (control[0]["k1"],control[0]["k2"]) == (10,4)
                assert row["matched"]["total_lines"] == row["total_lines"]
                assert row["matched"]["joint_pair_histogram_matched"]
        trajectories[method] = source_cases
        comparison[method] = dict(
            ef_mae=interval([x["absolute_error"] for x in rows]),
            rmse=float(np.sqrt(np.mean([x["absolute_error"]**2 for x in rows]))),
            mean_lines=float(np.mean([x["mean_lines"] for x in rows])),
            frame_weighted_lines=sum(x["total_lines"] for x in rows)/sum(x["frames"] for x in rows),
            psnr=float(np.mean([x["mean_psnr"] for x in rows])),
            ssim=float(np.mean([x["mean_ssim"] for x in rows])),
            floating_mae=float(np.mean([x["mean_mae"] for x in rows])),
            unobserved_mae=float(np.mean([np.mean([x["unobserved_mae"] for x in s]) for s in source_cases.values()])),
            switch_rate_mean=float(np.mean(switches)),
            warm_pair_histogram={f"{a}+{b}": n for (a,b),n in sorted(hist.items())},
            constant_warm_videos=sum(len(set((x["k1"],x["k2"]) for x in seq[1:])) == 1 for seq in source_cases.values()),
            matched_delta=interval([x["absolute_error"]-x["matched"]["absolute_error"] for x in rows]) if method != "E0" else None,
        )
        if method != "E0":
            comparison[method]["vs_e0_delta"] = interval([x["absolute_error"]-y["absolute_error"] for x,y in zip(rows,confirm["E0"])])
            delta = [x["absolute_error"]-x["matched"]["absolute_error"] for x in rows]
            comparison[method]["vs_matched_wins_ties_losses"] = [sum(x < -1e-8 for x in delta),sum(abs(x)<=1e-8 for x in delta),sum(x>1e-8 for x in delta)]
        timing[method] = dict(
            mean_video_s=float(np.mean([x["seconds"] for x in rows])),
            median_video_s=float(np.median([x["seconds"] for x in rows])),
            p95_video_s=float(np.quantile([x["seconds"] for x in rows],.95)),
            video_min_s=min(x["seconds"] for x in rows),video_max_s=max(x["seconds"] for x in rows),
            warm_frame_p50_s=float(np.median(frame_times)),warm_frame_p95_s=float(np.quantile(frame_times,.95)),
            cold_frame_p50_s=float(np.median(cold_times)),
            total_frames=sum(x["frames"] for x in rows),total_seconds=sum(x["seconds"] for x in rows),
            fps=sum(x["frames"] for x in rows)/sum(x["seconds"] for x in rows),
            total_perception_calls=sum(x["perception_calls"] for x in rows),
            mean_parts_s={k:float(np.mean([x[k] for x in rows])) for k in ["ef_selection_s","perception_s","controller_s","final_task_s","io_s"]},
        )
        # Reproduce the original report rather than substitute a new aggregation.
        point=next(x for x in summary["points"] if x["method"]==method and x["cohort"]=="confirmation")
        assert abs(point["absolute_error"]-comparison[method]["ef_mae"]["mean"]) < 1e-10
        if method != "E0":
            assert np.allclose(point["matched"]["paired_video_bootstrap_ci95"],comparison[method]["matched_delta"]["ci95"],atol=1e-10)

    for i, name in enumerate(case_ids):
        xs = {m: confirm[m][i] for m in confirm}
        row = dict(case=name, frames=xs["E0"]["frames"], ef_truth=xs["E0"]["ef_truth"], full_polar_error=xs["E0"]["full_input_absolute_error"], original_cartesian_error=xs["E0"]["original_input_absolute_error"])
        for m in ["E0","E1","E2"]:
            row.update({f"{m}_error":xs[m]["absolute_error"],f"{m}_lines":xs[m]["mean_lines"],f"{m}_switch_rate":xs[m]["budget_switch_rate"]})
            if m != "E0":row.update({f"{m}_control_error":xs[m]["matched"]["absolute_error"],f"{m}_minus_control":xs[m]["absolute_error"]-xs[m]["matched"]["absolute_error"]})
        cases.append(row)
    write_csv(out / "confirmation_cases.csv", cases)
    train, training_rows, checkpoint_states = {}, [], {}
    for method in ["E1","E2"]:
        all_updates = {}
        durations, checkpoints = {}, {}
        for endpoint in [50,100,200]:
            d=jobs/f"{method}_u{endpoint:04d}_train"
            info=loaded(d/"result.json");durations[str(endpoint)]=info["process_wall_s"]
            p=d/"console.log";source_hashes[str(p.relative_to(r)).replace("\\","/")]=digest(p)
            for line in p.read_text(encoding="utf-8").splitlines():
                try:x=json.loads(line)
                except ValueError:continue
                if x.get("event")=="train":
                    assert x["update"] not in all_updates
                    all_updates[x["update"]]=x
            p=d/"policy.npz";source_hashes[str(p.relative_to(r)).replace("\\","/")]=digest(p)
            with np.load(p,allow_pickle=False) as z:
                assert int(z["step"])==endpoint and all(np.isfinite(z[k]).all() for k in z.files)
                checkpoints[str(endpoint)]={k:z[k].copy() for k in z.files}
        assert sorted(all_updates)==list(range(1,201))
        xs=[all_updates[i] for i in range(1,201)]
        assert all(x["case"] in cohorts["train"] for x in xs)
        assert all(np.isfinite(x["gradient_norm"]) for x in xs)
        train[method]=dict(
            updates=200, unique_videos=len({x["case"] for x in xs}),frames_processed=sum(x["frames"] for x in xs),
            clip_frames_min=min(x["frames"] for x in xs),clip_frames_max=max(x["frames"] for x in xs),
            case_counts=dict(Counter(x["case"] for x in xs)),
            gradient_min=min(x["gradient_norm"] for x in xs),gradient_median=float(np.median([x["gradient_norm"] for x in xs])),gradient_max=max(x["gradient_norm"] for x in xs),
            gradient_clipped=sum(x["gradient_clipped"] for x in xs),update_p50_s=float(np.median([x["seconds"] for x in xs])),update_p95_s=float(np.quantile([x["seconds"] for x in xs],.95)),
            stage_seconds=durations,total_process_s=sum(durations.values()),
            head_change_50_to_200={h:float(np.sqrt(sum(np.sum((checkpoints['200'][k]-checkpoints['50'][k])**2) for k in checkpoints['200'] if k.startswith('params.'+h+'.')))) for h in ['first','second']},
            final_policy_stats=xs[-1]["policy_stats"],
            buckets=[dict(start=i+1,end=i+50,ef_error=float(np.mean([x['ef_absolute_error'] for x in xs[i:i+50]])),lines=float(np.mean([x['mean_lines'] for x in xs[i:i+50]]))) for i in [0,50,100,150]],
        )
        if method == 'E1':
            train[method].update(max_forward_drift=max(x['hard_replay_max_abs'] for x in xs),max_ad_primal_drift=max(x['gradient_linearization_primal_max_abs'] for x in xs),all_task_gradients_positive=all(all(v>0 for v in x['task_gradient_norms'].values()) for x in xs))
        for x in xs:
            training_rows.append(dict(method=method,update=x['update'],case=x['case'],frames=x['frames'],ef_error=x['ef_absolute_error'],lines=x['mean_lines'],gradient_norm=x['gradient_norm'],clipped=x['gradient_clipped'],seconds=x['seconds'],first_max_probability=x['policy_stats']['first']['mean_max_probability'],second_max_probability=x['policy_stats']['second']['mean_max_probability']))
        checkpoint_states[method]=xs
    assert [(x['case'],x['start'],x['frames'],x['action_seed']) for x in checkpoint_states['E1']]==[(x['case'],x['start'],x['frames'],x['action_seed']) for x in checkpoint_states['E2']]
    write_csv(out/'training_updates.csv',training_rows)
    development=[]
    names=cohorts['development'][:4]
    for method in ['E0','untrained','E1','E2']:
        for endpoint in ([0] if method in ['E0','untrained'] else [50,100,200]):
            job=('E0_10_4_s42_development' if method=='E0' else 'untrained_development' if method=='untrained' else f'{method}_u{endpoint:04d}_development')
            rows=[x for x in records(job) if x['case'] in names];assert len(rows)==4
            development.append(dict(method=method,updates=endpoint,ef_mae=float(np.mean([x['absolute_error'] for x in rows])),lines=float(np.mean([x['mean_lines'] for x in rows])),psnr=float(np.mean([x['mean_psnr'] for x in rows])),matched_ef_mae=float(np.mean([x['matched']['absolute_error'] for x in rows])) if all('matched' in x for x in rows) else None))
    write_csv(out/'development_common_four.csv',development)
    # Timing ledger is from archived job results, not estimated from a frame microbenchmark.
    stage=[dict(job=x['job'],seconds=read(jobs/x['job']/'result.json').get('process_wall_s',0)) for x in summary['ledger']]
    peak=max(read(jobs/x['job']/'result.json').get('process_ram_sample_peak_bytes',0) for x in summary['ledger'])
    timing['batch']=dict(wall_s=status['finished']-status['started'],stages=stage,ram_sample_peak_bytes=peak)
    references={k:float(np.mean([x[k] for x in confirm['E0']])) for k in ['full_input_absolute_error','original_input_absolute_error','conversion_prediction_shift']}
    info=dict(confirmation=comparison,timing=timing,training=train,common_four=development,full_input_references=references,cases=cases,
        protocol=dict(bootstrap_replicates=5000,bootstrap_seed=20261008,unit='video',seed_count=1,confirmation_cases=16,frames=sum(x['frames'] for x in confirm['E0'])),
        validation=dict(tasks_completed=18,train_dev_test_disjoint=True,same_training_stream=True,joint_histogram_control_exact=True,all_checkpoint_states_finite=True,e0_reused=loaded(r/'repair_lineage.json')['reused_e0_cases']))
    (out/'analysis.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    plot(out,confirm,comparison,development,trajectories)
    for name in ['config.json','manifest.json','identity.json','summary.json','status.json','plan.json']:
        source_hashes[name]=digest(r/name)
    import matplotlib
    sources=dict(source_directory=str(r),source_hashes=source_hashes,script_sha256=digest(Path(__file__)),output_hashes={p.name:digest(p) for p in out.iterdir() if p.suffix in ['.csv','.png','.pdf','.json'] and p.name!='provenance.json'},scope='post-run descriptive analysis; no exclusion, retraining or GPU experiment; no multiple-testing adjustment',software=dict(python=platform.python_version(),numpy=np.__version__,matplotlib=matplotlib.__version__),figures=dict(publisher='not specified; general research report',png_dpi=180,physical_width_inches={'confirmation_comparison':11.4,'common_four_checkpoints':11.4,'budget_trajectory_example':10.2},uncertainty='95% video bootstrap percentile intervals;5000 replicates;seed20261008;one training seed',aggregation='equal video; warm-frame histograms count all warm frames; fixed common four cases for checkpoints',trajectory_example='first locked confirmation case, not outcome selected'))
    (out/'provenance.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf-8',newline='\n')
    print(json.dumps({k:info[k] for k in ['confirmation','timing','training','common_four','full_input_references','validation']},ensure_ascii=False,indent=2))


def plot(out, confirm, stats, development, trajectories):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():font_manager.fontManager.addfont(str(font));name=font_manager.FontProperties(fname=str(font)).get_name()
    else:name='DejaVu Sans'
    colors={'E0':'#333333','E1':'#0072B2','E2':'#B54A00'}
    markers={'E0':'o','E1':'s','E2':'^'}
    plt.rcParams.update({'font.family':name,'font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,'figure.facecolor':'white'})
    def save(fig,stem):
        for ext in ['png','pdf']:fig.savefig(out/f'{stem}.{ext}',dpi=180,facecolor='white')
        plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11.4,4.5),layout='constrained')
    for m in colors:
        s=stats[m];v=s['ef_mae'];lo,hi=v['ci95'];axs[0].errorbar(s['mean_lines'],v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt=markers[m],color=colors[m],capsize=4,label=m,markersize=8)
    axs[0].set(xlabel='平均采集线数／帧',ylabel='EF MAE（百分点）',title='同一批16个独立测试视频',xlim=(0,28),ylim=(0,10));axs[0].legend()
    for i,m in enumerate(['E1','E2']):
        d=np.array([x['absolute_error']-x['matched']['absolute_error'] for x in confirm[m]]);v=stats[m]['matched_delta'];lo,hi=v['ci95']
        offsets=np.linspace(-.16,.16,len(d));axs[1].scatter(i+offsets,d,color=colors[m],marker=markers[m],alpha=.65,s=25)
        axs[1].errorbar(i+.28,v['mean'],yerr=[[v['mean']-lo],[hi-v['mean']]],fmt='D',color='black',capsize=4,label='均值及95%区间' if i==0 else None)
    axs[1].axhline(0,color='#777777',linestyle='--');axs[1].set(xticks=[0,1],xticklabels=['E1','E2'],ylabel='候选误差 − 同预算周期对照误差（百分点）',title='负值表示候选更好；每个点是一个视频');axs[1].legend(loc='upper right',fontsize=9)
    for ax in axs:ax.grid(alpha=.15)
    fig.suptitle('左为病例MAE区间，右为配对差值区间；95%病例bootstrap，5000次；单一种子')
    save(fig,'confirmation_comparison')
    fig,axs=plt.subplots(1,2,figsize=(11.4,4.2),layout='constrained')
    for m in ['E1','E2']:
        rows=[x for x in development if x['method']==m];zero=next(x for x in development if x['method']=='untrained');rows=[zero,*rows]
        axs[0].plot([x['updates'] for x in rows],[x['ef_mae'] for x in rows],marker=markers[m],linestyle='-' if m=='E1' else '--',color=colors[m],label=m)
        axs[1].plot([x['updates'] for x in rows],[x['lines'] for x in rows],marker=markers[m],linestyle='-' if m=='E1' else '--',color=colors[m],label=m)
    base=next(x for x in development if x['method']=='E0')
    for ax,key in zip(axs,['ef_mae','lines']):ax.axhline(base[key],color=colors['E0'],linestyle=':',label='E0 固定10＋4');ax.set_xlabel('累计更新次数');ax.set_xticks([0,50,100,200]);ax.grid(alpha=.15);ax.legend(loc='lower left')
    axs[0].set(ylabel='EF MAE（百分点）',ylim=(0,9));axs[1].set(ylabel='平均线数／帧',ylim=(0,28));fig.suptitle('始终使用相同4个开发病例；连线仅表示检查点顺序，不是连续测量')
    save(fig,'common_four_checkpoints')
    fig,axs=plt.subplots(2,1,figsize=(10.2,5.8),sharex=True,layout='constrained')
    # Predeclared display choice: first locked confirmation case; not best/worst-case selection.
    case=confirm['E0'][0]['case']
    for ax,m in zip(axs,['E1','E2']):
        seq=trajectories[m][case];ax.step(np.arange(len(seq)),[x['k1'] for x in seq],where='post',color=colors[m],label='第一阶段',linestyle='-')
        ax.step(np.arange(len(seq)),[x['k2'] for x in seq],where='post',color='#555555',label='第二阶段',linestyle='--');ax.set(ylabel=f'{m} 扫描线数',ylim=(0,16));ax.grid(alpha=.15);ax.legend(loc='lower left')
    axs[-1].set_xlabel('视频帧索引（首帧固定10＋4）');fig.suptitle(f'锁定测试名单的第一个病例：{case}\n展示预算确实变化；未提供心动周期标注，不能证明变化发生在关键时刻',fontsize=10)
    save(fig,'budget_trajectory_example')


if __name__ == '__main__':main()
