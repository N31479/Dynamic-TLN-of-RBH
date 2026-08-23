#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';RES.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
rows=[];wins=[]
for d in sorted((ROOT/'scan').glob('q*')):
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
rows=sorted(rows,key=lambda x:(x['ell_over_ell_ext'],x['omega_M']))
with (RES/'bardeen_axial_metric_dynamic_tln.csv').open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with (RES/'bardeen_axial_metric_windows.csv').open('w',newline='') as h:
    w=csv.DictWriter(h,fieldnames=list(wins[0]));w.writeheader();w.writerows(wins)

freqs=[.002,.004,.006];ratios=sorted(set(r['ell_over_ell_ext'] for r in rows if r['omega_M'] in freqs))
static=[next(r['k_static'] for r in rows if r['ell_over_ell_ext']==x) for x in ratios]
fig,ax=plt.subplots(figsize=(7.2,5.0));ax.plot(ratios,static,'--',lw=1.8,label=r'$M\omega=0$')
for om in freqs:
    rr=sorted([r for r in rows if r['omega_M']==om],key=lambda z:z['ell_over_ell_ext'])
    ax.plot([r['ell_over_ell_ext'] for r in rr],[r['k_metric_real'] for r in rr],marker='o',ms=3.2,lw=1.25,label=rf'$M\omega={om:.3f}$')
ax.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$k_{20}^{\rm axial}(\omega)$',title='Bardeen axial metric dynamical TLN');ax.grid(alpha=.22);ax.legend(frameon=False)
ins=ax.inset_axes([.17,.18,.48,.36])
for om in freqs:
    rr=sorted([r for r in rows if r['omega_M']==om],key=lambda z:z['ell_over_ell_ext'])
    ins.plot([r['ell_over_ell_ext'] for r in rr],[1e4*r['delta_k_real'] for r in rr],marker='o',ms=2.2,lw=1)
ins.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$10^4[k(\omega)-k(0)]$',title='Dynamical correction');ins.grid(alpha=.2);ins.tick_params(labelsize=7)
fig.tight_layout();fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_ell_ratio.png',dpi=300);fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_ell_ratio.pdf');plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,5.0))
for ratio in [.5,.8,.95]:
    rr=sorted([r for r in rows if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
    x=[0.]+[r['omega_M'] for r in rr];y=[rr[0]['k_static']]+[r['k_metric_real'] for r in rr]
    ax.plot(x,y,marker='o',ms=3.4,lw=1.3,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$')
ax.set(xlabel=r'$M\omega$',ylabel=r'$k_{20}^{\rm axial}(\omega)$',title='Bardeen axial metric dynamical TLN versus frequency');ax.grid(alpha=.22);ax.legend(frameon=False)
fig.tight_layout();fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_frequency.png',dpi=300);fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_frequency.pdf');plt.close(fig)
summary={'definition':'Direct metric h0 response/source ratio in the pure gravitational-source sector; independently recovered static baseline; no Schwarzschild subtraction','maximum_window_residual':max(r['window_max_residual'] for r in rows),'maximum_em_source_abs':max(r['em_source_abs'] for r in wins),'maximum_grav_source_error':max(r['grav_source_error'] for r in wins),'frequencies':[.002,.004,.006],'frequency_scan_ratios':[.5,.8,.95]}
(RES/'bardeen_axial_metric_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(summary)
