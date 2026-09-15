"""Regenerate publication figures from the committed summary tables.

No bespoke colors or global plotting styles are used. Every figure is a
separate matplotlib plot. Run after experiments.analyze_results.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'; OUT.mkdir(exist_ok=True)
LABELS={'task_task':'Task design / task stop','full_task':'Full design / task stop',
        'full_full':'Full design / full stop','task_full':'Task design / full stop',
        'uniform_task':'Uniform / task stop','entropy_task':'Entropy / task stop'}

def read(name):
    with (ROOT/'results'/name).open() as f:return list(csv.DictReader(f))

def save(fig,name):
    fig.tight_layout()
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

def main():
    rows=read('exact_summary.csv')
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    for method,marker in zip(['task_task','full_task','full_full','uniform_task','entropy_task'],['o','s','^','D','v']):
        r=sorted([r for r in rows if r['method']==method],key=lambda r:float(r['gamma']))
        x=np.array([float(t['gamma']) for t in r]);y=np.array([float(t['restricted_mean']) for t in r])
        err=np.array([[float(t['restricted_mean'])-float(t['ci_low']) for t in r],
                      [float(t['ci_high'])-float(t['restricted_mean']) for t in r]])
        ax.errorbar(x,y,yerr=err,marker=marker,capsize=3,label=LABELS[method])
    ax.set(xscale='log',yscale='log',xlabel=r'Nuisance-effect separation $\gamma$',
           ylabel='Calibration transitions (mean)',title='Control can be grounded before all effects are identified')
    ax.set_xticks([.01,.02,.04,.08],labels=['0.01','0.02','0.04','0.08'])
    ax.legend(fontsize=8);save(fig,'query_complexity')
    rows=read('confidence_summary.csv');fig,ax=plt.subplots(figsize=(6.4,3.7))
    for method,marker in [('task_task','o'),('entropy_task','s')]:
        r=sorted([r for r in rows if r['method']==method],key=lambda r:-float(r['delta']))
        x=np.array([-np.log10(float(t['delta'])) for t in r]);y=np.array([float(t['restricted_mean']) for t in r])
        err=np.array([[float(t['restricted_mean'])-float(t['ci_low']) for t in r],
                      [float(t['ci_high'])-float(t['restricted_mean']) for t in r]])
        ax.errorbar(x,y,yerr=err,marker=marker,capsize=3,label=LABELS[method])
    ax.set(xlabel=r'Required confidence, $\log_{10}(1/\delta)$',ylabel='Calibration transitions (mean)',
           title='Task-directed design becomes advantageous at stricter confidence')
    ax.legend(fontsize=9);save(fig,'confidence_scaling')
    rows=read('information_rates.csv');fig,ax=plt.subplots(figsize=(6.4,3.7))
    for target,marker,label in [('task','o','Task grounding'),('full','s','Complete mapping recovery')]:
        r=[r for r in rows if r['target']==target]
        ax.plot([float(t['gamma']) for t in r],[float(t['inverse_rate']) for t in r],marker=marker,label=label)
    ax.set(xscale='log',yscale='log',xlabel=r'Nuisance-effect separation $\gamma$',ylabel=r'Characteristic time $1/D^*$',
           title='Theoretical complexity: relevant versus irrelevant permutation cycles')
    ax.legend();save(fig,'information_geometry')
    rows=read('learned_summary.csv');fig,ax=plt.subplots(figsize=(6.4,3.7))
    for robust,marker,label in [('False','o','Plug-in likelihood (uncorrected)'),('True','s','Robust likelihood (oracle radius)')]:
        r=sorted([r for r in rows if r['robust']==robust],key=lambda r:int(r['source_n_per_effect_context']))
        ax.plot([int(t['source_n_per_effect_context']) for t in r],[int(t['certified'])/int(t['runs']) for t in r],marker=marker,label=label)
    ax.set(xscale='log',xlabel='Passive samples per anonymous effect',ylabel='Fraction certified by 3,000 queries',
           ylim=(-.03,1.05),title='Robustness requires enough accuracy in the pretrained effects')
    ax.legend(fontsize=9);save(fig,'learned_robustness')
    rows=read('mismatch_summary.csv');fig,ax=plt.subplots(figsize=(6.4,3.7))
    for robust,marker,label in [('False','o','Uncorrected: incorrect certificates'),('True','s','Robust: incorrect certificates')]:
        r=sorted([r for r in rows if r['robust']==robust],key=lambda r:float(r['alpha']))
        ax.plot([float(t['alpha']) for t in r],[int(t['wrong_certificates'])/int(t['runs']) for t in r],marker=marker,label=label)
    r=sorted([r for r in rows if r['robust']=='True'],key=lambda r:float(r['alpha']))
    ax.plot([float(t['alpha']) for t in r],[int(t['certified'])/int(t['runs']) for t in r],linestyle='--',marker='^',label='Robust: all certificates')
    ax.set(xlabel=r'Context-misalignment strength $\alpha$',ylabel='Fraction of all runs',ylim=(-.03,1.05),
           title='More calibration does not repair an inconsistent pretrained model')
    ax.legend(fontsize=8);save(fig,'misspecification')
    print('Generated five separate figures (PDF, SVG, PNG) in',OUT)

if __name__=='__main__':main()
