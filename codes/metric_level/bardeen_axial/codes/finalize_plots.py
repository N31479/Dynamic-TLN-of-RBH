#!/usr/bin/env python3
from __future__ import annotations
import csv,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RES=ROOT/'results';FIG=ROOT/'figures'
rows=list(csv.DictReader((RES/'bardeen_axial_metric_dynamic_tln.csv').open()))
for r in rows:
 for k,v in list(r.items()):
  try:r[k]=float(v)
  except:pass

fig,ax=plt.subplots(figsize=(7.4,5.1))
for ratio in [.5,.8,.95]:
 rr=sorted([r for r in rows if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
 x=np.array([0.]+[r['omega_M'] for r in rr]);y=np.array([rr[0]['k_static']]+[r['k_metric_real'] for r in rr])
 ax.plot(x,y,marker='o',ms=3.5,lw=1.3,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$')
ax.set(xlabel=r'$M\omega$',ylabel=r'$k_{20}^{\rm axial}(\omega)$',title='Bardeen axial metric dynamical TLN versus frequency');ax.grid(alpha=.22);ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=3,fontsize=8)
ins=ax.inset_axes([.17,.17,.47,.38])
for ratio in [.5,.8,.95]:
 rr=sorted([r for r in rows if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
 ins.plot([r['omega_M'] for r in rr],[1e4*r['delta_k_real'] for r in rr],marker='o',ms=2.3,lw=1)
ins.set(xlabel=r'$M\omega$',ylabel=r'$10^4\Delta k$',title='Conservative correction');ins.grid(alpha=.2);ins.tick_params(labelsize=7)
fig.tight_layout();fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_frequency.png',dpi=300);fig.savefig(FIG/'bardeen_axial_metric_dynamic_tln_vs_frequency.pdf');plt.close(fig)

broad={}
for ratio in [.6,.9,.97]:
 p=RES/f'bardeen_axial_broad_q{int(round(100*ratio)):03d}.csv';rr=list(csv.DictReader(p.open()))
 std=sorted([r for r in rr if int(r['strict_controls'])==0],key=lambda z:float(z['omega_M']))
 broad[ratio]={'omega':np.array([float(r['omega_M']) for r in std]),'abs':np.array([float(r['delta_Rgg_abs']) for r in std])}
ring=list(csv.DictReader((RES/'bardeen_axial_ringdown.csv').open()))
for r in ring:
 for k,v in list(r.items()):
  if k in {'qnm_resonance_aligned','controls_preserve_alignment'}:r[k]=v.lower()=='true'
  else:
   try:r[k]=float(v)
   except:pass
fig,axes=plt.subplots(1,2,figsize=(10.7,4.4))
for ratio in [.6,.9,.97]:
 b=broad[ratio];axes[0].plot(b['omega'],b['abs'],'o-',ms=3,lw=1.2,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$')
axes[0].set(xlabel=r'$M\omega$',ylabel=r'$|\mathcal{R}_{gg}^{\rm axial}|$',title='Broad canonical axial response');axes[0].grid(alpha=.22);axes[0].legend(frameon=False,fontsize=8)
x=np.arange(3);peaks=np.array([r['real_axis_peak_M'] for r in ring]);modes=np.array([r['ringdown_frequency_M'] for r in ring]);err=np.array([r['maximum_control_shift'] for r in ring])
axes[1].plot(x,peaks,'D-',label='real-axis maximum');axes[1].errorbar(x,modes,yerr=err,fmt='o-',capsize=3,label='axial ringdown frequency')
axes[1].set_xticks(x,[f'{r:.2f}' for r in [.6,.9,.97]]);axes[1].set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$M\omega$',title='QNM-resonance alignment');axes[1].grid(alpha=.22);axes[1].legend(frameon=False,fontsize=8)
fig.tight_layout();fig.savefig(FIG/'bardeen_axial_resonance_test.png',dpi=300);fig.savefig(FIG/'bardeen_axial_resonance_test.pdf');plt.close(fig)

fits={}
for ratio in [.5,.8,.95]:
 rr=sorted([r for r in rows if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
 w=np.array([r['omega_M'] for r in rr]);dr=np.array([r['delta_k_real'] for r in rr]);di=np.array([r['delta_k_imag'] for r in rr])
 a2=np.linalg.lstsq((w**2)[:,None],dr,rcond=None)[0][0];a1=np.linalg.lstsq(w[:,None],di,rcond=None)[0][0]
 fits[str(ratio)]={'conservative_omega2_coefficient':float(a2),'dissipative_omega_coefficient':float(a1),'max_relative_conservative_fit_error':float(np.max(abs(dr-a2*w**2))/max(np.max(abs(dr)),1e-30))}
(RES/'bardeen_axial_low_frequency_fits.json').write_text(json.dumps(fits,indent=2)+'\n')
print(json.dumps(fits,indent=2))
