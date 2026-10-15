"""Recompute contrasts and t/TOST/Holm independently of experiment analysis helpers."""
from pathlib import Path
from collections import defaultdict
import csv
import gzip
import json
import math
import os
import sys

os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np

import argparse
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path)
args=ap.parse_args()
study=Path(__file__).resolve().parents[1]
out=args.output or study.parents[1]/'reproduced/independent-stage2-statistics'
if out.exists() and any(out.iterdir()):
    raise SystemExit('Choose a fresh output directory; retained audit results are not overwritten.')
out.mkdir(parents=True,exist_ok=True)
root=study
units=json.loads((study/'provenance/validation/raw_unit_checks.json').read_text())
rep=json.loads((study/'prep/heldout_final/heldout_analysis.json').read_text())
idx={(u['session'],u['block'],u['arm'],u['length_s']):u for u in units}
diffs=[]

def same(a,b,key,tol=1e-10):
    if not math.isclose(a,b,rel_tol=tol,abs_tol=tol):
        diffs.append({'key':key,'independent':a,'reported':b})

def sf(t):
    return .5-t/(2*math.sqrt(t*t+2))

def tquantile(p):
    v=2*p-1
    return v*math.sqrt(2/(1-v*v))

def holm(values):
    previous=0
    answer={}
    for i,(k,p) in enumerate(sorted(values.items(),key=lambda kv:kv[1])):
        previous=max(previous,min(1.,(len(values)-i)*p))
        answer[k]=previous
    return answer

def ratio(s,b,a,L,ref,RL,est,metric='J'):
    x=idx[(s,b,a,L)][metric][est]
    y=idx[(s,b,ref,RL)][metric][est]
    return None if x is None or y is None else math.log(x/y)

all_contrasts=[]
for est in ('primary','raw_counter','nominal_period','sampled_instant'):
    fam={'HD1':{},'HD3':{},'HD4':{'C/A@40':{}}}
    if est!='sampled_instant':
        fam['HD2']={'C':{}}
    for s in (1,2,3):
        for a in 'ABC':
            for L in (2.5,10.,20.):
                fam['HD1'].setdefault(f'{a}|{L:g}',{})[s]=[x for b in (1,2)
                    if (x:=ratio(s,b,a,L,a,40.,est)) is not None]
        for L in (10.,20.,40.):
            fam['HD3'].setdefault(f'{L:g}',{})[s]=[x for b in (1,2)
                if (x:=ratio(s,b,'B',L,'A',L,est)) is not None]
        fam['HD4']['C/A@40'][s]=[ratio(s,b,'C',40.,'A',40.,est) for b in (1,2)]
        if est!='sampled_instant':
            coefs=[]
            for b in (1,2):
                sel=[u for u in units if u['session']==s and u['block']==b and u['E'][est] is not None]
                X=np.array([[1.,1/u['T_region_s'],float(u['arm']=='C')] for u in sel])
                y=np.array([math.log(u['E']['sampled_instant']/u['E'][est]) for u in sel])
                beta=np.linalg.solve(X.T@X,X.T@y)
                coefs.append(float(beta[2]))
            fam['HD2']['C'][s]=coefs
    for f,members in fam.items():
        d=math.log(1.005 if f=='HD3' else 1.004 if f=='HD4' else 1.003)
        stats={}
        for member,blocks in members.items():
            means=[sum(blocks[s])/len(blocks[s]) for s in (1,2,3)]
            mean=sum(means)/3
            sd=math.sqrt(sum((x-mean)**2 for x in means)/2)
            se=sd/math.sqrt(3)
            ci95=[mean-tquantile(.975)*se,mean+tquantile(.975)*se]
            ci90=[mean-tquantile(.95)*se,mean+tquantile(.95)*se]
            p=2*sf(abs(mean/se))
            beyond=min(1.,2*min(sf((mean-d)/se),1-sf((mean+d)/se)))
            tost=max(sf((mean+d)/se),1-sf((mean-d)/se))
            stats[member]={'mean':mean,'sd':sd,'se':se,'session_means':means,'p':p,'beyond':beyond,'tost':tost,
                           'ci95':ci95,'ci90':ci90,'pct':100*math.expm1(mean),
                           'ci95_pct':[100*math.expm1(x) for x in ci95],'halfwidth_log_pct':tquantile(.975)*se*100}
        hp=holm({m:x['p'] for m,x in stats.items()})
        hb=holm({m:x['beyond'] for m,x in stats.items()})
        for member,x in stats.items():
            sm=rep['estimators'][est]['summaries'][f][member]
            e=rep['estimators'][est][f][member]
            for k in ('mean','sd','se'):
                same(x[k],sm[k],f'{est}/{f}/{member}/{k}')
            for i,s in enumerate((1,2,3)):
                same(x['session_means'][i],sm['session_means'][str(s)],f'{est}/{f}/{member}/session{s}')
            for k in ('ci95','ci90'):
                for i in (0,1):
                    same(x[k][i],sm[k][i],f'{est}/{f}/{member}/{k}/{i}')
            for a,b in ((x['p'],e['difference']['p']), (hp[member],e['difference']['p_holm']),
                        (hb[member],e['difference']['p_beyond_margin_holm']), (x['tost'],e['equivalence']['p_tost'])):
                same(a,b,f'{est}/{f}/{member}/p')
            all_contrasts.append({'estimator':est,'family':f,'member':member,**x,'p_holm':hp[member],
                                 'p_beyond_holm':hb[member],'frozen_verdict':e['verdict']})

# Recorded per-GPU pre-region temperatures are recomputed directly from telemetry, before assessing R3.
thermal=[]
for s in (1,2,3):
    for L in (2.5,10.,20.,40.):
        for arm in 'BC':
            ds={}
            for g in (4,5):
                def avg(a):
                    return sum(next(p for p in idx[(s,b,a,L)]['per_gpu'] if p['gpu']==g)['temp_precomputed'] for b in (1,2))/2
                ds[g]=avg(arm)-avg('A')
            k=f'{s}|{L:g}|{arm}-A'
            r=rep['estimators']['primary']['r3'][k]
            for g,d in ds.items():
                same(d,r['diff_c_by_gpu'][str(g)],f'R3/{k}/GPU{g}')
            thermal.append({'key':k,'differences':ds,'frozen_ok':r['ok'],
                            'rounding_tolerance_ok':all(abs(x)<=.5+1e-12 for x in ds.values())})

# Diagnostic timing only: an exploratory latency comparison is not a predeclared Stage 2 finding.
latency=[]
for L in (10.,20.,40.):
    ms=[sum(math.log(idx[(s,b,'C',L)]['t_gpu_s']/idx[(s,b,'A',L)]['t_gpu_s']) for b in (1,2))/2 for s in (1,2,3)]
    latency.append({'length_s':L,'session_latency_C_over_A_pct':[100*math.expm1(x) for x in ms],
                    'mean_pct':100*math.expm1(sum(ms)/3),'role':'EXPLORATORY audit diagnostic; not added to frozen decisions'})

tables=[]  # Numerical-table byte checks belong to reproduce_public.py, not this independent formula check.
result={'independent_contrasts':all_contrasts,'numeric_differences':diffs,'thermal_checks':thermal,
        'exploratory_latency':latency,'table_reproduction':tables,
        'notes':'Closed-form t distribution with df=2, independent manual raw energy; original functions not called.'}
(out/'independent_statistics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'contrasts':len(all_contrasts),'numeric_differences':len(diffs),
                  'primary_key_results':[x for x in all_contrasts if x['estimator']=='primary' and x['family'] in ('HD3','HD4')],
                  'table_reproduction':tables,'exploratory_latency':latency},indent=2),flush=True)
assert not diffs,diffs[:3]
