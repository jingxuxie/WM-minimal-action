"""Aggregate every recorded run; deterministic bootstrap confidence intervals.

Learned-model intervals resample pretraining seeds (clusters), not individual
calibration episodes. All query means are restricted means at the stated cap.
"""
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
LABELS={'task_task':'Task design / task stop','full_task':'Full design / task stop',
        'full_full':'Full design / full stop','task_full':'Task design / full stop',
        'uniform_task':'Uniform / task stop','entropy_task':'Entropy / task stop'}


def read(name):
    with (ROOT/'results'/name).open() as f:
        return list(csv.DictReader(f))


def boot_mean(values,rng,repetitions=4000):
    values=np.asarray(values,dtype=float)
    samples=values[rng.integers(len(values),size=(repetitions,len(values)))].mean(axis=1)
    lo,hi=np.quantile(samples,[.025,.975])
    return float(values.mean()),float(lo),float(hi)


def main():
    rng=np.random.default_rng(92319)
    settings=[('exact_runs.csv',['gamma','method']),('confidence_runs.csv',['delta','method']),
              ('learned_runs.csv',['source_n_per_effect_context','robust']),
              ('mismatch_runs.csv',['alpha','robust']),('negative_control_runs.csv',['method'])]
    summaries={}
    for filename,fields in settings:
        data=read(filename);groups={}
        for r in data:
            key=tuple(r[f] for f in fields);groups.setdefault(key,[]).append(r)
        output=[]
        for key,rows in groups.items():
            times=[float(r['transitions']) for r in rows]
            if filename=='learned_runs.csv':
                clusters={}
                for r in rows:
                    clusters.setdefault(r['pretraining_seed'],[]).append(float(r['transitions']))
                mean,lo,hi=boot_mean([np.mean(v) for v in clusters.values()],rng)
            else:
                mean,lo,hi=boot_mean(times,rng)
            cert=[r for r in rows if r['certified']=='True']
            item=dict(zip(fields,key))|{'runs':len(rows),'restricted_mean':mean,'ci_low':lo,'ci_high':hi,
                'cap':int(rows[0]['cap']),'certified':len(cert),'censored':len(rows)-len(cert),
                'wrong_certificates':sum(r['task_correct']=='False' for r in cert),
                'coverage_failures':sum(r['coverage_failure']=='True' for r in rows),
                'mean_final_regret':float(np.mean([float(r['regret']) for r in rows])),
                'mean_confidence_size':float(np.mean([float(r['confidence_size']) for r in rows])),
                'mean_diagnostic_fraction':float(np.mean([float(r['diagnostic_fraction']) for r in rows]))}
            output.append(item)
        summaries[filename]=output
        path=ROOT/'results'/filename.replace('_runs.csv','_summary.csv')
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    (ROOT/'results'/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    data=summaries['exact_runs.csv']
    tex=['% Automatically generated from raw experiment results. Do not edit numbers manually.',
         r'\begin{tabular}{lrrrr}',r'\toprule',
         r'Query design / stop & $\gamma=.01$ & $.02$ & $.04$ & $.08$ \\',r'\midrule']
    for method in LABELS:
        selected={float(r['gamma']):r for r in data if r['method']==method}
        values=[]
        for g in [.01,.02,.04,.08]:
            r=selected[g];suffix=r'^{\dagger}' if r['censored'] else ''
            values.append('$'+f"{r['restricted_mean']:.1f}"+suffix+'$')
        tex.append(LABELS[method]+' & '+' & '.join(values)+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}']
    (ROOT/'paper'/'table_exact.tex').write_text('\n'.join(tex)+'\n')
    notes=['# Measured results','',
           'All values below come from executed synthetic CPU experiments. Means include right-censored runs at their caps. Bootstrap intervals are 95% percentile intervals (4,000 resamples).',
           '', '## Exact-model nuisance sweep','',
           '| gamma | method | capped mean [95% CI] | certified / runs | incorrect task certificates |',
           '|---:|---|---:|---:|---:|']
    for r in data:
        notes.append(f"| {r['gamma']} | {LABELS[r['method']]} | {r['restricted_mean']:.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}] | {r['certified']}/{r['runs']} | {r['wrong_certificates']} |")
    notes+=['','## Interpretation','',
      'The roughly 100x comparison at gamma=0.01 is against full mapping recovery, a harder objective; it is not a 100x improvement over all task-directed baselines. Entropy/task uses fewer samples than Task/task at ordinary confidence. At delta=1e-10, Task/task uses fewer samples than Entropy/task. The all-effects-required negative control makes Task/task and Full/full identical sample by sample.',
      '', '## Learned models and misspecification','',
      'Learned-model runs use anonymous pure demonstrator IDs and categorical likelihood estimates. This is not a neural/video representation experiment. Robust radii use simulator truth as an explicit diagnostic oracle; a deployable radius estimator is not evaluated. The task deployment kernels remain known. Robust intervals are conservative and often abstain. Uncorrected certification can become systematically incorrect under context misalignment.',
      '', '## Data accounting','',
      'There are '+str(sum(sum(r['runs'] for r in v) for v in summaries.values()))+' recorded calibration runs across all five suites. Learned-model runs contain 20 independent pretraining seeds per sample size and 24 calibration episodes per pretrained model. Exact and confidence settings use 200 independent calibration replicas; paired random streams are shared between methods within each setting.']
    (ROOT/'docs'/'RESULTS.md').write_text('\n'.join(notes)+'\n')
    print(json.dumps({'total_recorded_runs':sum(sum(r['runs'] for r in v) for v in summaries.values()),
                      'gamma_001':data[:6]},indent=2))

if __name__=='__main__':main()
