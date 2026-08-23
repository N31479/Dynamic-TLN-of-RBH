#!/usr/bin/env python3

from __future__ import annotations
import csv,json,sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master

@dataclass(frozen=True)
class Controls:
 spacing:float=.10;outer_radius:float=180.;horizon_offset:float=1e-8;final_time:float=150.;courant:float=.45;pulse_offset:float=12.;pulse_width:float=3.;extraction_offset:float=25.

def grid(metric,horizon,c):
 def event(_,r):return float(r[0]-c.outer_radius)
 event.terminal=True;event.direction=1
 sol=solve_ivp(lambda _,r:[float(metric(r[0]))],(0,1500),[horizon+c.horizon_offset],events=event,dense_output=True,rtol=1e-11,atol=1e-13,max_step=.1)
 L=float(sol.t_events[0][0]);n=int(L/c.spacing)+1;x=np.linspace(0,L,n);return x,sol.sol(x)[0]

def evolve(model,c):
 x,r=grid(model.f,master.outer_horizon(model),c);dx=x[1]-x[0];V=master.potential_matrix(model,'polar',r,2);W=model.f(r)[:,None,None]*V
 peak=np.argmax(np.linalg.eigvalsh(W)[:,-1]);center=x[peak]+c.pulse_offset;extract=np.argmin(abs(x-(x[peak]+c.extraction_offset)))
 u=np.zeros((2,len(x)));u[0]=np.exp(-((x-center)/c.pulse_width)**2)
 def acc(z):
  a=np.zeros_like(z);a[:,1:-1]=(z[:,2:]-2*z[:,1:-1]+z[:,:-2])/dx**2-np.einsum('nij,jn->in',W[1:-1],z[:,1:-1]);return a
 dt=c.courant*dx;steps=int(c.final_time/dt);dt=c.final_time/steps;prev=u.copy();cur=u+.5*dt**2*acc(u);ab=(dt-dx)/(dt+dx);times=[];sig=[]
 for n in range(1,steps):
  nxt=2*cur-prev+dt**2*acc(cur);nxt[:,0]=cur[:,1]+ab*(nxt[:,1]-cur[:,0]);nxt[:,-1]=cur[:,-2]+ab*(nxt[:,-2]-cur[:,-1]);prev,cur=cur,nxt
  if n%2==0:times.append((n+1)*dt);sig.append(cur[0,extract])
 return np.asarray(times),np.asarray(sig),float(r[peak])

def fit(t,y,start=60,stop=120):
 mask=(t>=start)&(t<=stop);tt=t[mask]-start;data=y[mask];scale=max(np.max(abs(data)),1e-10);best=None
 def res(p):return p[0]*np.exp(-p[1]*tt)*np.cos(p[2]*tt+p[3])-data
 for guess in np.linspace(.25,.65,9):
  z=least_squares(res,[data[0],.09,guess,0],bounds=([-2*scale,0,.15,-20],[2*scale,.4,.9,20]),max_nfev=4000);rr=np.linalg.norm(z.fun)/np.linalg.norm(data)
  if best is None or rr<best[0]:best=(rr,z.x)
 rr,p=best;return dict(frequency_M=float(p[2]),damping_M=float(p[1]),amplitude=float(p[0]),phase=float(p[3]),residual=float(rr),start=start,stop=stop)

def broad_data():
 rows=list(csv.DictReader((ROOT/'results'/'all_models_broad_frequency.csv').open()));out={}
 for model in ['hayward','fan_wang']:
  for ratio in [.6,.9,.97]:
   standard=sorted([r for r in rows if r['model']==model and float(r['ell_over_ell_ext'])==ratio and int(r['strict_controls'])==0],key=lambda r:float(r['omega_M']))
   strict=sorted([r for r in rows if r['model']==model and float(r['ell_over_ell_ext'])==ratio and int(r['strict_controls'])==1],key=lambda r:float(r['omega_M']))
   om=np.array([float(r['omega_M']) for r in standard]);val=np.array([float(r['delta_Rgg_real'])+1j*float(r['delta_Rgg_imag']) for r in standard]);vals=np.array([float(r['delta_Rgg_real'])+1j*float(r['delta_Rgg_imag']) for r in strict]);i=int(np.argmax(abs(val)))
   out[(model,ratio)]=dict(omega=om,value=val,strict=vals,peak=float(om[i]),peak_abs=float(abs(val[i])),strict_change=float(abs(vals[i]-val[i])/max(abs(vals[i]),1e-30)))
 return out

def run():
 broad=broad_data();rows=[];convergence=[];waveforms={}
 variants=[('spacing',.08,Controls(spacing=.08)),('spacing',.12,Controls(spacing=.12)),('outer_radius',220.,Controls(outer_radius=220.)),('horizon_offset',2e-8,Controls(horizon_offset=2e-8))]
 for model_name in ['hayward','fan_wang']:
  for ratio in [.6,.9,.97]:
   model=master.build_model(model_name,ratio*master.extremal_charge(model_name,1.));t,y,rpeak=evolve(model,Controls());f=fit(t,y);z=complex(f['frequency_M'],-f['damping_M']);shifts=[];control_modes=[z]
   for name,value,c in variants:
    tv,yv,_=evolve(model,c);fv=fit(tv,yv);zv=complex(fv['frequency_M'],-fv['damping_M']);shift=abs(zv-z);shifts.append(shift);control_modes.append(zv);convergence.append(dict(model=model_name,ell_over_ell_ext=ratio,control=name,control_value=value,frequency_M=zv.real,damping_M=-zv.imag,complex_shift=shift,fit_residual=fv['residual']))
   for start,stop in [(55,115),(65,125),(70,140)]:
    fv=fit(t,y,start,stop);zv=complex(fv['frequency_M'],-fv['damping_M']);shift=abs(zv-z);shifts.append(shift);control_modes.append(zv);convergence.append(dict(model=model_name,ell_over_ell_ext=ratio,control='fit_window',control_value=start,frequency_M=zv.real,damping_M=-zv.imag,complex_shift=shift,fit_residual=fv['residual']))
   b=broad[(model_name,ratio)];sep=abs(z.real-b['peak']);detuning=sep/(-z.imag);robust=max(abs(b['peak']-mode.real)/(-mode.imag) for mode in control_modes);aligned=detuning<=1.
   rows.append(dict(model=model_name,ell_over_ell_ext=ratio,real_axis_peak_M=b['peak'],peak_abs=b['peak_abs'],strict_peak_relative_change=b['strict_change'],ringdown_frequency_M=z.real,ringdown_damping_M=-z.imag,quality_factor=z.real/(-2*z.imag),peak_ringdown_separation=sep,linewidth_detuning=detuning,robust_linewidth_detuning=robust,qnm_resonance_aligned=aligned,controls_preserve_alignment=((robust<=1.)==aligned),maximum_control_shift=max(shifts),fit_residual=f['residual'],potential_peak_radius_M=rpeak));waveforms[(model_name,ratio)]=(t,y,f)
 results=ROOT/'results';figs=ROOT/'figures'
 for fn,data in [('resonance_test.csv',rows),('resonance_convergence.csv',convergence)]:
  with (results/fn).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 summary={'criterion':'A real-frequency maximum is QNM-resonance aligned when its nominal detuning from the fitted real QNM frequency is no larger than one damping width. Control fits test whether the classification is preserved. This is not a pole identification.','cases':rows,'all_qnm_resonance_aligned':all(r['qnm_resonance_aligned'] for r in rows)};(results/'resonance_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 for model_name in ['hayward','fan_wang']:
  fig,axes=plt.subplots(1,2,figsize=(10.6,4.4));colors=plt.cm.viridis(np.linspace(.15,.85,3));model_rows=[r for r in rows if r['model']==model_name]
  for color,ratio in zip(colors,[.6,.9,.97]):
   b=broad[(model_name,ratio)];axes[0].plot(b['omega'],abs(b['value']),'o-',ms=2.8,lw=1.2,color=color,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$');axes[0].axvline(b['peak'],color=color,alpha=.25,lw=.9)
  axes[0].set(xlabel=r'$M\omega$',ylabel=r'$|\mathcal{R}_{gg}^{\rm polar}|$',title='Broad canonical response');axes[0].grid(alpha=.22);axes[0].legend(frameon=False,fontsize=8)
  x=np.arange(3);peaks=np.array([r['real_axis_peak_M'] for r in model_rows]);rings=np.array([r['ringdown_frequency_M'] for r in model_rows]);errs=np.array([r['maximum_control_shift'] for r in model_rows])
  axes[1].plot(x,peaks,'D-',label='real-axis maximum');axes[1].errorbar(x,rings,yerr=errs,fmt='o-',capsize=3,label='ringdown frequency');axes[1].set_xticks(x,[f'{r:.2f}' for r in [.6,.9,.97]]);axes[1].set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$M\omega$',title='QNM-resonance alignment');axes[1].grid(alpha=.22);axes[1].legend(frameon=False,fontsize=8)
  fig.tight_layout();base=figs/f'{model_name}_polar_resonance_test';fig.savefig(base.with_suffix('.png'),dpi=300);fig.savefig(base.with_suffix('.pdf'));plt.close(fig)
 return rows
if __name__=='__main__':
 for r in run():print(r)
