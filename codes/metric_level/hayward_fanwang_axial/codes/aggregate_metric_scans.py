#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';RES.mkdir(exist_ok=True)
rows=[];wins=[]
for d in sorted((ROOT/'scan').glob('*_q*')):
    if (d/'response.csv').exists(): rows+=list(csv.DictReader((d/'response.csv').open()))
    if (d/'windows.csv').exists(): wins+=list(csv.DictReader((d/'windows.csv').open()))
for r in rows:
    for k,v in list(r.items()):
        try:r[k]=float(v)
        except:pass
for r in wins:
    for k,v in list(r.items()):
        try:r[k]=float(v)
        except:pass
for r in rows:
    r['delta_k_real']=r['raw_delta_real']; r['delta_k_imag']=r['raw_delta_imag']
    r['k_metric_real']=r['k_unsubtracted_real']; r['k_metric_imag']=r['k_unsubtracted_imag']
rows=sorted(rows,key=lambda r:(r['model'],r['ell_over_ell_ext'],r['omega_M']))
wins=sorted(wins,key=lambda r:(r['model'],r['ell_over_ell_ext'],r['omega_M'],r['match_radius']))
for fn,data in [('hayward_fanwang_axial_metric_dynamic_tln.csv',rows),('hayward_fanwang_axial_metric_windows.csv',wins)]:
    with (RES/fn).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
summary={'definition':'Direct reconstructed h0 response/source ratio in the pure gravitational-source sector; physical odd EM source nu=Phi/(4 sqrt(L_F)); independently recovered static baseline; no Schwarzschild subtraction','controlled_range':'Momega<=0.006','models':{}}
for model in ('hayward','fan_wang'):
    mr=[r for r in rows if r['model']==model];mw=[r for r in wins if r['model']==model]
    summary['models'][model]={'maximum_window_residual':max(r['window_max_residual'] for r in mr),'maximum_em_source_abs':max(r['em_source_abs'] for r in mw),'maximum_grav_source_error':max(r['grav_source_error'] for r in mw),'maximum_source_transform_condition':max(r['source_transform_condition'] for r in mr),'maximum_response_transform_condition':max(r['response_transform_condition'] for r in mr)}
(RES/'axial_metric_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
