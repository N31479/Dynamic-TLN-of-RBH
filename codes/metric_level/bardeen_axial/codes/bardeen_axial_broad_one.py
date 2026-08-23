#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
ROOT=Path(__file__).resolve().parents[1]

def calc(t):
 ratio,w,strict=t;model=master.build_model('bardeen',ratio*master.extremal_charge('bardeen',1.),1.)
 c=master.SolverControls(horizon_offset=1e-5 if strict else 2e-5,match_radius=20 if strict else 18,asymptotic_cycles=45 if strict else 30,minimum_outer_radius=380 if strict else 240,rtol=3e-10 if strict else 1e-9,atol=3e-12 if strict else 1e-11,max_step_phase=.2 if strict else .3,max_step_radius=5.)
 z=master.canonical_gravitational_response(model,'axial',w,c)
 return dict(ell_over_ell_ext=ratio,omega_M=w,strict_controls=int(strict),delta_Rgg_real=z.real,delta_Rgg_imag=z.imag,delta_Rgg_abs=abs(z))

def main():
 p=argparse.ArgumentParser();p.add_argument('--ratio',type=float,required=True);a=p.parse_args();ratio=a.ratio
 coarse=[.10,.20,.30,.40,.50,.60,.70,.80]
 with ProcessPoolExecutor(max_workers=4) as ex:rows=list(ex.map(calc,[(ratio,w,False) for w in coarse]))
 peak=max(rows,key=lambda x:x['delta_Rgg_abs'])['omega_M'];ref=[max(.1,peak-.075),max(.1,peak-.0375),peak,min(.8,peak+.0375),min(.8,peak+.075)];ref=sorted(set(round(x,4) for x in ref)-set(coarse))
 if ref:
  with ProcessPoolExecutor(max_workers=4) as ex:rows+=list(ex.map(calc,[(ratio,w,False) for w in ref]))
 p0=max(rows,key=lambda x:x['delta_Rgg_abs']);strict=calc((ratio,p0['omega_M'],True));rows.append(strict);rows=sorted(rows,key=lambda x:(x['strict_controls'],x['omega_M']))
 out=ROOT/'results'/f'bardeen_axial_broad_q{int(round(100*ratio)):03d}.csv'
 with out.open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps({'ratio':ratio,'peak':p0,'strict':strict},indent=2))
if __name__=='__main__':main()
