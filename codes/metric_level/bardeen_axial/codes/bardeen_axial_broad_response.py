#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
ROOT=Path(__file__).resolve().parents[1]

def one(args):
    ratio,omega,strict=args
    model=master.build_model('bardeen',ratio*master.extremal_charge('bardeen',1.),1.)
    c=master.SolverControls(horizon_offset=1e-5 if strict else 2e-5,match_radius=20. if strict else 18.,asymptotic_cycles=50. if strict else 35.,minimum_outer_radius=450. if strict else 300.,rtol=3e-10 if strict else 1e-9,atol=3e-12 if strict else 1e-11,max_step_phase=.18 if strict else .25,max_step_radius=5.)
    z=master.canonical_gravitational_response(model,'axial',omega,c)
    return dict(ell_over_ell_ext=ratio,omega_M=omega,strict_controls=int(strict),delta_Rgg_real=z.real,delta_Rgg_imag=z.imag,delta_Rgg_abs=abs(z))

def main():
    ratios=[.6,.9,.97];coarse=np.round(np.arange(.1,.801,.05),10)
    tasks=[(r,float(w),False) for r in ratios for w in coarse]
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs=[ex.submit(one,t) for t in tasks]
        for f in as_completed(futs):rows.append(f.result())
    
    for ratio in ratios:
        rr=[x for x in rows if x['ell_over_ell_ext']==ratio and not x['strict_controls']];peak=max(rr,key=lambda x:x['delta_Rgg_abs'])['omega_M']
        grid=np.round(np.arange(max(.1,peak-.05),min(.8,peak+.05)+1e-9,.0125),10)
        existing={x['omega_M'] for x in rr}
        tasks=[(ratio,float(w),False) for w in grid if float(w) not in existing]
        with ProcessPoolExecutor(max_workers=3) as ex:
            for f in as_completed([ex.submit(one,t) for t in tasks]):rows.append(f.result())
    
    strict_tasks=[]
    for ratio in ratios:
        rr=[x for x in rows if x['ell_over_ell_ext']==ratio and not x['strict_controls']];peak=max(rr,key=lambda x:x['delta_Rgg_abs'])['omega_M'];strict_tasks.append((ratio,peak,True))
    with ProcessPoolExecutor(max_workers=3) as ex:
        for f in as_completed([ex.submit(one,t) for t in strict_tasks]):rows.append(f.result())
    rows=sorted(rows,key=lambda x:(x['ell_over_ell_ext'],x['strict_controls'],x['omega_M']))
    out=ROOT/'results'/'bardeen_axial_broad_response.csv'
    with out.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for ratio in ratios:
        rr=[x for x in rows if x['ell_over_ell_ext']==ratio and not x['strict_controls']];p=max(rr,key=lambda x:x['delta_Rgg_abs']);s=[x for x in rows if x['ell_over_ell_ext']==ratio and x['strict_controls']][0]
        summary.append(dict(ell_over_ell_ext=ratio,peak_omega_M=p['omega_M'],peak_abs=p['delta_Rgg_abs'],relative_peak_response_change=abs(complex(s['delta_Rgg_real'],s['delta_Rgg_imag'])-complex(p['delta_Rgg_real'],p['delta_Rgg_imag']))/max(s['delta_Rgg_abs'],1e-30)))
    (ROOT/'results'/'bardeen_axial_broad_response_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
if __name__=='__main__':main()
