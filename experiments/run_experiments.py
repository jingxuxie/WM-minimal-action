#!/usr/bin/env python3
"""Reproduce exact-model sweeps, confidence scaling, and model-error tests.
Run from repository root. CPU only; no API, GPU, or downloaded dataset.
"""
from __future__ import annotations
import argparse, csv, json, time, platform, sys
from pathlib import Path
import numpy as np
import scipy
from wm_grounding.environments import diagnostic_chain, learned_anchor_kernels, exact_log_radius
from wm_grounding.core import Problem, cycle_design
from wm_grounding.simulation import simulate, METHODS
ROOT = Path(__file__).resolve().parents[1]

def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)

def annotate(rows, **fields):
    return [dict(fields,**row) for row in rows]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repetitions',type=int,default=200)
    ap.add_argument('--cap',type=int,default=50000)
    ap.add_argument('--pretraining-seeds',type=int,default=20)
    ap.add_argument('--pretraining-repetitions',type=int,default=24)
    ap.add_argument('--suite',choices=['all','exact','confidence','learned','mismatch','structure'],default='all')
    ap.add_argument('--output',type=Path,default=ROOT/'results')
    args=ap.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    checkpoints=(1,2,5,10,20,40,80,160,320,640,1280,2560,5120,10000,20000,50000)
    metadata={'argv':sys.argv,'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
              'platform':platform.platform(),'config':vars(args)|{'output':str(out)},
              'interpretation':'CPU synthetic reset-query experiments; cap is right censoring; exact deployment values, not rollout estimates.'}
    if args.suite in ('all','exact'):
        rows,curves=[],[]
        for gamma in [.01,.02,.04,.08]:
            p=diagnostic_chain(gamma)
            for method in METHODS:
                t=time.monotonic()
                r,c=simulate(p,method,args.repetitions,args.cap,seed=1701,checkpoints=checkpoints)
                rows += annotate(r,gamma=gamma,delta=.05,required='0;1',contexts=9,horizon=4)
                curves += annotate(c,gamma=gamma,method=method,delta=.05)
                print('exact',gamma,method,'mean',round(np.mean([x['transitions'] for x in r]),2),
                      'cert',np.mean([x['certified'] for x in r]),'seconds',round(time.monotonic()-t,2),flush=True)
                write_csv(out/'exact_runs.csv',rows); write_csv(out/'exact_curves.csv',curves)
    if args.suite in ('all','confidence'):
        rows=[]; p=diagnostic_chain(.02,contexts=2)
        for delta in [.1,.01,1e-3,1e-6,1e-10]:
            for method in ['task_task','entropy_task','full_task']:
                r,_=simulate(p,method,args.repetitions,args.cap,delta=delta,seed=2718)
                rows+=annotate(r,gamma=.02,delta=delta,contexts=2,horizon=4)
                print('confidence',delta,method,'mean',round(np.mean([x['transitions'] for x in r]),2),flush=True)
        write_csv(out/'confidence_runs.csv',rows)
    if args.suite in ('all','learned'):
        rows,models=[],[]; true=diagnostic_chain(.02,contexts=1)
        for n_source in [256,1024,4096,16384,65536]:
            for k in range(args.pretraining_seeds):
                p_hat=learned_anchor_kernels(true,n_source,np.random.default_rng(31000+k))
                nominal=Problem(p_hat,required=true.required)
                radius=exact_log_radius(true.kernels,p_hat)
                models.append({'source_n_per_effect_context':n_source,'pretraining_seed':31000+k,
                               'oracle_log_radius':radius.tolist(),'kernels':p_hat.tolist()})
                for robust in [False,True]:
                    r,_=simulate(nominal,'task_task',args.pretraining_repetitions,3000,seed=41000+k,
                                 actual_kernels=true.kernels,log_radius=radius if robust else None)
                    rows+=annotate(r,source_n_per_effect_context=n_source,
                                   total_passive_samples=n_source*4,pretraining_seed=31000+k,
                                   robust=robust,oracle_log_radius=float(radius.max()))
            print('learned',n_source,flush=True)
            write_csv(out/'learned_runs.csv',rows)
        (out/'learned_models.json').write_text(json.dumps(models,indent=2)+'\n')
    if args.suite in ('all','mismatch'):
        rows=[]; true=diagnostic_chain(.02,contexts=1)
        for alpha in [0,.1,.3,.5,.7,1.]:
            swapped=true.kernels[:,[1,0,2,3],:]
            learned=(1-alpha)*true.kernels+alpha*swapped
            nominal=Problem(learned,required=true.required)
            radius=exact_log_radius(true.kernels,learned)
            for robust in [False,True]:
                r,_=simulate(nominal,'task_task',args.repetitions,3000,seed=5772,
                             actual_kernels=true.kernels,log_radius=radius if robust else None)
                rows+=annotate(r,alpha=alpha,robust=robust,oracle_log_radius=float(radius.max()))
                print('mismatch',alpha,robust,'cert',np.mean([x['certified'] for x in r]),
                      'task_correct',np.mean([x['task_correct'] for x in r]),flush=True)
        write_csv(out/'mismatch_runs.csv',rows)
    if args.suite in ('all','structure'):
        rows=[]
        for gamma in np.geomspace(.001,.08,15):
            p=diagnostic_chain(float(gamma),contexts=2)
            for target in ['task','full']:
                w,d=p.designs(target); required=p.required if target=='task' else tuple(range(p.m))
                cw,cd,it=cycle_design(p.kernels,p.perms[0],required)
                assert np.isclose(cd,d[0],rtol=1e-6,atol=1e-9)
                rows.append({'gamma':gamma,'target':target,'D_star':d[0],'inverse_rate':1/d[0],
                             'cycle_D_star':cd,'cycle_iterations':it,
                             **{f'w_action_{j}':w[0].reshape(p.contexts,p.m)[:,j].sum() for j in range(p.m)}})
        write_csv(out/'information_rates.csv',rows)
        rows=[]; p=diagnostic_chain(.04,contexts=2,required=(0,1,2,3),horizon=4)
        for method in ['task_task','full_full','entropy_task','uniform_task']:
            r,_=simulate(p,method,args.repetitions,args.cap,seed=8119)
            rows+=annotate(r,required='0;1;2;3',gamma=.04)
        write_csv(out/'negative_control_runs.csv',rows)
        comp=[]; rates=[]
        for scale in [1.,.8,.6,.4]:
            base=diagnostic_chain(.04,contexts=1)
            kernels=scale*base.kernels+(1-scale)/base.outcomes
            problem=Problem(kernels); _,d=problem.designs('task'); rates.append(d[0])
            comp.append({'components':len(rates),'new_component_strength':scale,
                         'new_component_D_star':d[0],
                         'sum_inverse_rates':sum(1/r for r in rates),
                         'joint_D_star':1/sum(1/r for r in rates)})
        write_csv(out/'component_rates.csv',comp)
    metadata['elapsed_seconds']=time.monotonic()-start
    (out/f'metadata_{args.suite}.json').write_text(json.dumps(metadata,indent=2,default=str)+'\n')
    print('finished',round(metadata['elapsed_seconds'],2),'seconds',flush=True)

if __name__=='__main__':
    main()
