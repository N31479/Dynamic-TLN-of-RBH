#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
ROOT=Path(__file__).resolve().parents[1]

def controls(strict=False):
    return master.SolverControls(horizon_offset=1e-5 if strict else 2e-5,match_radius=20. if strict else 18.,asymptotic_cycles=50. if strict else 35.,minimum_outer_radius=450. if strict else 300.,rtol=3e-10 if strict else 1e-9,atol=3e-12 if strict else 1e-11,max_step_phase=.18 if strict else .25,max_step_radius=5.)

def one(t):
    model_name,ratio,omega,strict=t
    model=master.build_model(model_name,ratio*master.extremal_charge(model_name,1.),1.)
    z=master.canonical_gravitational_response(model,'axial',omega,controls(strict))
    return dict(model=model_name,ell_over_ell_ext=ratio,omega_M=omega,strict_controls=int(strict),delta_Rgg_real=z.real,delta_Rgg_imag=z.imag,delta_Rgg_abs=abs(z))

def main():
    models=['hayward','fan_wang'];ratios=[.6,.9,.97];coarse=np.round(np.arange(.1,.801,.025),10)
    rows=[];tasks=[(m,r,float(w),False) for m in models for r in ratios for w in coarse]
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(one,t) for t in tasks]):rows.append(f.result())
    
    refine=[]
    for m in models:
        for r in ratios:
            rr=[x for x in rows if x['model']==m and x['ell_over_ell_ext']==r and not x['strict_controls']];p=max(rr,key=lambda x:x['delta_Rgg_abs'])['omega_M']
            existing={x['omega_M'] for x in rr}
            for w in np.round(np.arange(max(.1,p-.025),min(.8,p+.025)+1e-10,.00625),10):
                if float(w) not in existing:refine.append((m,r,float(w),False))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(one,t) for t in refine]):rows.append(f.result())
    strict=[]
    for m in models:
        for r in ratios:
            rr=[x for x in rows if x['model']==m and x['ell_over_ell_ext']==r and not x['strict_controls']];p=max(rr,key=lambda x:x['delta_Rgg_abs'])['omega_M'];strict.append((m,r,p,True))
    with ProcessPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(one,t) for t in strict]):rows.append(f.result())
    rows=sorted(rows,key=lambda x:(x['model'],x['ell_over_ell_ext'],x['strict_controls'],x['omega_M']))
    out=ROOT/'results'/'hayward_fanwang_axial_broad_response.csv'
    with out.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for m in models:
        for r in ratios:
            rr=[x for x in rows if x['model']==m and x['ell_over_ell_ext']==r and not x['strict_controls']];p=max(rr,key=lambda x:x['delta_Rgg_abs']);s=[x for x in rows if x['model']==m and x['ell_over_ell_ext']==r and x['strict_controls']][0]
            zp=complex(p['delta_Rgg_real'],p['delta_Rgg_imag']);zs=complex(s['delta_Rgg_real'],s['delta_Rgg_imag'])
            summary.append(dict(model=m,ell_over_ell_ext=r,peak_omega_M=p['omega_M'],peak_abs=p['delta_Rgg_abs'],relative_peak_response_change=abs(zs-zp)/max(abs(zs),1e-30)))
    (ROOT/'results'/'hayward_fanwang_axial_broad_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
