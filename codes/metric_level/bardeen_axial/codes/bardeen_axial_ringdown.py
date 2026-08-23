#!/usr/bin/env python3

from __future__ import annotations
import csv,json
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
ROOT=Path(__file__).resolve().parents[1]
@dataclass(frozen=True)
class C: spacing:float=.10;outer:float=180.;eps:float=1e-8;tf:float=150.;courant:float=.45;pulse:float=12.;width:float=3.;extract:float=25.
def grid(metric,rh,c):
 def ev(_,r):return float(r[0]-c.outer)
 ev.terminal=True;ev.direction=1
 s=solve_ivp(lambda _,r:[float(metric(r[0]))],(0,1500),[rh+c.eps],events=ev,dense_output=True,rtol=1e-11,atol=1e-13,max_step=.1)
 L=float(s.t_events[0][0]);n=int(L/c.spacing)+1;x=np.linspace(0,L,n);return x,s.sol(x)[0]
def evolve(metric,potential,rh,c=C()):
 x,r=grid(metric,rh,c);dx=x[1]-x[0];V=np.asarray(potential(r),float)
 if V.ndim==1:V=V[:,None,None]
 W=np.asarray(metric(r))[:,None,None]*V;pk=int(np.argmax(np.linalg.eigvalsh(W)[:,-1]));center=x[pk]+c.pulse;ex=int(np.argmin(abs(x-(x[pk]+c.extract))))
 u=np.zeros((V.shape[1],len(x)));u[0]=np.exp(-((x-center)/c.width)**2)
 def acc(z):
  a=np.zeros_like(z);a[:,1:-1]=(z[:,2:]-2*z[:,1:-1]+z[:,:-2])/dx**2-np.einsum('nij,jn->in',W[1:-1],z[:,1:-1]);return a
 dt=c.courant*dx;steps=int(c.tf/dt);dt=c.tf/steps;prev=u.copy();cur=u+.5*dt**2*acc(u);ab=(dt-dx)/(dt+dx);tt=[];yy=[]
 for n in range(1,steps):
  nxt=2*cur-prev+dt**2*acc(cur);nxt[:,0]=cur[:,1]+ab*(nxt[:,1]-cur[:,0]);nxt[:,-1]=cur[:,-2]+ab*(nxt[:,-2]-cur[:,-1]);prev,cur=cur,nxt
  if n%2==0:tt.append((n+1)*dt);yy.append(cur[0,ex])
 return np.asarray(tt),np.asarray(yy),float(r[pk])
def fit(t,y,start=60,stop=120):
 m=(t>=start)&(t<=stop);tt=t[m]-start;d=y[m];scale=max(np.max(abs(d)),1e-10);best=None
 def res(p):return p[0]*np.exp(-p[1]*tt)*np.cos(p[2]*tt+p[3])-d
 for g in np.linspace(.25,.70,10):
  z=least_squares(res,[d[0],.09,g,0],bounds=([-2*scale,0,.15,-20],[2*scale,.45,.9,20]),max_nfev=4000);rr=np.linalg.norm(z.fun)/np.linalg.norm(d)
  if best is None or rr<best[0]:best=(rr,z.x)
 rr,p=best;return dict(frequency_M=float(p[2]),damping_M=float(p[1]),residual=float(rr),start=start,stop=stop)
def problem(ratio):
 model=master.build_model('bardeen',ratio*master.extremal_charge('bardeen',1.));return model.f,lambda r:master.potential_matrix(model,'axial',r,2),master.outer_horizon(model)
def schw():return lambda r:1-2/np.asarray(r),lambda r:master.schwarzschild_potential('axial',r,1.,2),2.
def peaks():
 rows=json.loads((ROOT/'results'/'bardeen_axial_broad_response_summary.json').read_text())
 return {float(row['ell_over_ell_ext']):float(row['peak_omega_M']) for row in rows}
def main():
 
 t,y,_=evolve(*schw(),C());sf=fit(t,y,65,140);ref=complex(.37367168,-.08896232);sc=complex(sf['frequency_M'],-sf['damping_M']);serr=abs(sc-ref)/abs(ref)
 rows=[];conv=[];variants=[('spacing',.08,C(spacing=.08)),('spacing',.12,C(spacing=.12)),('outer',220.,C(outer=220.)),('eps',2e-8,C(eps=2e-8))]
 for ratio,peak in sorted(peaks().items()):
  t,y,rpk=evolve(*problem(ratio),C());f=fit(t,y);z=complex(f['frequency_M'],-f['damping_M']);sh=[];control_modes=[z]
  for name,val,c in variants:
   tv,yv,_=evolve(*problem(ratio),c);fv=fit(tv,yv);zz=complex(fv['frequency_M'],-fv['damping_M']);sh.append(abs(zz-z));control_modes.append(zz);conv.append(dict(ell_over_ell_ext=ratio,control=name,value=val,frequency_M=zz.real,damping_M=-zz.imag,shift=abs(zz-z),residual=fv['residual']))
  for st,sp in [(55,115),(65,125),(70,140)]:
   fv=fit(t,y,st,sp);zz=complex(fv['frequency_M'],-fv['damping_M']);sh.append(abs(zz-z));control_modes.append(zz);conv.append(dict(ell_over_ell_ext=ratio,control='fit_window',value=st,frequency_M=zz.real,damping_M=-zz.imag,shift=abs(zz-z),residual=fv['residual']))
  sep=abs(z.real-peak);detuning=sep/(-z.imag);robust=max(abs(peak-mode.real)/(-mode.imag) for mode in control_modes);aligned=bool(detuning<=1.);rows.append(dict(ell_over_ell_ext=ratio,real_axis_peak_M=peak,ringdown_frequency_M=z.real,ringdown_damping_M=-z.imag,quality_factor=z.real/(-2*z.imag),peak_ringdown_separation=sep,linewidth_detuning=detuning,robust_linewidth_detuning=robust,qnm_resonance_aligned=aligned,controls_preserve_alignment=bool((robust<=1.)==aligned),maximum_control_shift=max(sh),fit_residual=f['residual'],potential_peak_radius_M=rpk,schwarzschild_relative_error=serr))
 for fn,data in [('bardeen_axial_ringdown.csv',rows),('bardeen_axial_ringdown_convergence.csv',conv)]:
  with (ROOT/'results'/fn).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 summary={'schwarzschild':{'computed_frequency_M':sc.real,'computed_damping_M':-sc.imag,'relative_error':serr},'criterion':'A real-frequency maximum is QNM-resonance aligned when its nominal detuning from the fitted real QNM frequency is no larger than one damping width. Control fits test whether the classification is preserved. This is not a pole identification.','cases':rows,'any_qnm_resonance_aligned':any(r['qnm_resonance_aligned'] for r in rows)};(ROOT/'results'/'bardeen_axial_resonance_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
