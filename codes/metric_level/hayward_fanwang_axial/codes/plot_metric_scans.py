#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures';FIG.mkdir(exist_ok=True)
rows=list(csv.DictReader((RES/'hayward_fanwang_axial_metric_dynamic_tln.csv').open()))
for r in rows:
    for k,v in list(r.items()):
        try:r[k]=float(v)
        except:pass
labels={'hayward':'Hayward','fan_wang':'Fan--Wang'}
for model in ('hayward','fan_wang'):
    mr=[r for r in rows if r['model']==model]
    freqs=[.002,.004,.006];ratios=sorted(set(r['ell_over_ell_ext'] for r in mr if r['omega_M'] in freqs))
    static=[]
    for x in ratios:static.append(next(r['k_static'] for r in mr if r['ell_over_ell_ext']==x))
    fig,ax=plt.subplots(figsize=(7.2,5.0));ax.plot(ratios,static,'--',lw=1.8,label=r'$M\omega=0$')
    for om in freqs:
        rr=sorted([r for r in mr if abs(r['omega_M']-om)<1e-12],key=lambda z:z['ell_over_ell_ext'])
        ax.plot([r['ell_over_ell_ext'] for r in rr],[r['k_metric_real'] for r in rr],marker='o',ms=3.2,lw=1.25,label=rf'$M\omega={om:.3f}$')
    ax.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$k_{20}^{\rm axial}(\omega)$',title=f"{labels[model]} axial metric dynamical TLN");ax.grid(alpha=.22);ax.legend(frameon=False)
    ins=ax.inset_axes([.16,.17,.49,.36])
    scale=1e4
    for om in freqs:
        rr=sorted([r for r in mr if abs(r['omega_M']-om)<1e-12],key=lambda z:z['ell_over_ell_ext'])
        ins.plot([r['ell_over_ell_ext'] for r in rr],[scale*r['delta_k_real'] for r in rr],marker='o',ms=2.1,lw=1)
    ins.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$10^4\Delta k$',title='Dynamical correction');ins.grid(alpha=.2);ins.tick_params(labelsize=7)
    fig.tight_layout();stem=f'{model}_axial_metric_dynamic_tln_vs_ell_ratio';fig.savefig(FIG/f'{stem}.png',dpi=300);fig.savefig(FIG/f'{stem}.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,5.0))
    for ratio in (.5,.8,.95):
        rr=sorted([r for r in mr if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
        x=[0.]+[r['omega_M'] for r in rr];y=[rr[0]['k_static']]+[r['k_metric_real'] for r in rr]
        ax.plot(x,y,marker='o',ms=3.4,lw=1.3,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$')
    ax.set(xlabel=r'$M\omega$',ylabel=r'$k_{20}^{\rm axial}(\omega)$',title=f"{labels[model]} axial metric dynamical TLN versus frequency");ax.grid(alpha=.22);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=3,fontsize=8)
    ins=ax.inset_axes([.17,.17,.47,.38])
    for ratio in (.5,.8,.95):
        rr=sorted([r for r in mr if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
        ins.plot([r['omega_M'] for r in rr],[1e4*r['delta_k_real'] for r in rr],marker='o',ms=2.3,lw=1)
    ins.set(xlabel=r'$M\omega$',ylabel=r'$10^4\Delta k$',title='Conservative correction');ins.grid(alpha=.2);ins.tick_params(labelsize=7)
    fig.tight_layout();stem=f'{model}_axial_metric_dynamic_tln_vs_frequency';fig.savefig(FIG/f'{stem}.png',dpi=300);fig.savefig(FIG/f'{stem}.pdf');plt.close(fig)

fits={}
for model in ('hayward','fan_wang'):
    fits[model]={}
    for ratio in (.5,.8,.95):
        rr=sorted([r for r in rows if r['model']==model and abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
        x=np.array([r['omega_M'] for r in rr]);re=np.array([r['delta_k_real'] for r in rr]);im=np.array([r['delta_k_imag'] for r in rr])
        c2=float(np.dot(x*x,re)/np.dot(x*x,x*x));c1=float(np.dot(x,im)/np.dot(x,x));fits[model][str(ratio)]={'conservative_c2':c2,'dissipative_c1':c1}
(RES/'axial_low_frequency_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
