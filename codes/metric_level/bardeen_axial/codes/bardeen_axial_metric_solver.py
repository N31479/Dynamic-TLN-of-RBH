#!/usr/bin/env python3

from __future__ import annotations
import csv,json,argparse
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
from bardeen_axial_nearzone_basis import build_modes,evaluate

SQRTLAM=2.0

def direct_static_fit(model):
    x=model.charge/model.mass
    return (-5.7638546264096995*x**4-1.0095588588541715*x**6
            -0.016395640985604535*x**8-0.4595953178610712*x**10)

def build_model(q,mass=1.):
    if q>0:return master.build_model('bardeen',q,mass)
    
    def mf(r):r=np.asarray(r,float);return mass+0*r
    def L(r):r=np.asarray(r,float);return 0*r
    def LF(r):r=np.asarray(r,float);return 7.5*mass/r
    def f(r):r=np.asarray(r,float);return 1-2*mass/r
    def fp(r):r=np.asarray(r,float);return 2*mass/r**2
    def kap(r):r=np.asarray(r,float);return -2+0*r
    def dm(r):r=np.asarray(r,float);return -0.25/r**2+1.5*mass/r**3
    def dp(r):r=np.asarray(r,float);return 6.25/r**2-13.5*mass/r**3
    return master.Model('bardeen',mass,0.,master.extremal_charge('bardeen',mass),f,fp,mf,L,LF,kap,dm,dp)

def reconstruct_h0(model,radii,values,derivatives):
    rr=np.asarray(radii,float)
    return model.f(rr)[:,None]/SQRTLAM*(values[:,0,:]+rr[:,None]*derivatives[:,0,:])

def horizon_basis(model,omega,radius,horizon_offset=2e-6,rtol=5e-10,atol=5e-12):
    rh=master.outer_horizon(model);start=rh+horizon_offset;fp_h=float(model.fp(rh));Vh=master.potential_matrix(model,'axial',rh,2)
    first=Vh/(fp_h-2j*omega);U0=np.eye(2,dtype=complex)+horizon_offset*first;Up0=-1j*omega*U0/float(model.f(start))+first
    y0=np.concatenate([U0.ravel(),Up0.ravel()])
    def rhs(r,y):
        U=y[:4].reshape(2,2);Up=y[4:].reshape(2,2);f=float(model.f(r));fp=float(model.fp(r));V=master.potential_matrix(model,'axial',r,2)
        return np.concatenate([Up.ravel(),(-(fp/f)*Up-((omega**2*np.eye(2)-f*V)@U)/f**2).ravel()])
    sol=solve_ivp(rhs,(start,radius),y0,method='DOP853',rtol=rtol,atol=atol,max_step=min(.25,.10/max(abs(omega),1e-6)))
    if not sol.success:raise RuntimeError(sol.message)
    y=sol.y[:,-1];return y[:4].reshape(2,2),y[4:].reshape(2,2)

def _constfit(r,y):
    r=np.asarray(r,float);X=np.column_stack([np.ones_like(r),1/r,1/r**2]);s=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/s,np.asarray(y).real,rcond=1e-13)[0]/s;ci=np.linalg.lstsq(X/s,np.asarray(y).imag,rcond=1e-13)[0]/s
    return complex(cr[0],ci[0])

def physical_transform(model,modes):
    rr=np.array([60.,80.,100.,120.,160.]);vals=np.zeros((len(rr),2,4),complex);ders=np.zeros_like(vals)
    for j,(e,c) in enumerate(modes):
        v,d=evaluate(c,e,0.,rr);vals[:,:,j]=v;ders[:,:,j]=d
    h0=reconstruct_h0(model,rr,vals,ders);exps=[e for e,_ in modes]
    gs=_constfit(rr,h0[:,0]/rr**3);gr=_constfit(rr,h0[:,2]*rr**2)
    pes=-exps[1]/2;per=-exps[3]/2
    es=_constfit(rr,vals[:,1,1]/rr**pes);er=_constfit(rr,vals[:,1,3]/rr**per)
    Ps=np.array([[gs,0],[0,es]],complex);Pr=np.array([[gr,0],[0,er]],complex)
    return np.linalg.inv(Ps),np.linalg.inv(Pr),Ps,Pr

def basis_data(modes,transform,radius,omega):
    val=np.zeros((2,4),complex);der=np.zeros((2,4),complex)
    for j,(e,c) in enumerate(modes):
        v,d=evaluate(c,e,omega,[radius]);val[:,j]=v[0];der[:,j]=d[0]
    Ts,Tr=transform
    return np.column_stack([val[:,:2]@Ts,val[:,2:]@Tr]),np.column_stack([der[:,:2]@Ts,der[:,2:]@Tr])

def response_coeff(model,modes,transform,omega,radius,controls):
    hv,hd=horizon_basis(model,omega,radius,**controls);av,ad=basis_data(modes,transform,radius,omega)
    B=np.vstack([av,ad]);H=np.vstack([hv,hd]);col=np.linalg.norm(B,axis=0);balanced=B/col
    C=np.linalg.solve(balanced,H)/col[:,None]
    weights=np.linalg.solve(C[:2,:],np.array([1.,0.],complex));amp=C@weights
    return amp[2],amp[3],np.linalg.cond(balanced),amp[0],amp[1]

def extrapolate(radii,values,omega,omega_order=2):
    r=np.asarray(radii,float);y=np.asarray(values,complex);p=2*(omega_order+1)
    X=np.column_stack([np.ones_like(r),(omega*r)**p,1/r**2]);s=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/s,y.real,rcond=1e-13)[0]/s;ci=np.linalg.lstsq(X/s,y.imag,rcond=1e-13)[0]/s
    pred=X@cr+1j*(X@ci);return complex(cr[0],ci[0]),float(np.sqrt(np.mean(abs(y-pred)**2))),pred

def solve_ratio(ratio,frequencies,match_radii=(10.,12.,14.,16.),series_order=22,omega_order=2,log_order=5,mass=1.,controls=None):
    if controls is None:controls=dict(horizon_offset=2e-6,rtol=5e-10,atol=5e-12)
    q=ratio*master.extremal_charge('bardeen',mass);model=build_model(q,mass)
    modes=build_modes(q,series_order,omega_order,log_order,mass);Ts,Tr,Ps,Pr=physical_transform(model,modes);transform=(Ts,Tr)
    static={r:response_coeff(model,modes,transform,0.,r,controls)[0] for r in match_radii}
    rows=[];windows=[]
    for omega in frequencies:
        vals=[]
        for r in match_radii:
            c,em,cond,gs,es=response_coeff(model,modes,transform,omega,r,controls);dc=c-static[r];vals.append(dc)
            windows.append(dict(ell_over_ell_ext=ratio,ell=q,omega_M=omega,match_radius=r,response_real=c.real,response_imag=c.imag,delta_real=dc.real,delta_imag=dc.imag,em_response_real=em.real,em_response_imag=em.imag,grav_source_error=abs(gs-1),em_source_abs=abs(es),condition=cond))
        dc,rms,pred=extrapolate(match_radii,vals,omega,omega_order);spread=float(max(abs(np.asarray(vals)-pred)))
        kstatic=0. if ratio==0 else direct_static_fit(model)
        rows.append(dict(ell_over_ell_ext=ratio,ell=q,omega_M=omega,k_static=kstatic,raw_delta_real=dc.real,raw_delta_imag=dc.imag,k_unsubtracted_real=kstatic+dc.real,k_unsubtracted_imag=dc.imag,window_fit_rms=rms,window_max_residual=spread,maximum_omega_r=max(match_radii)*abs(omega),source_transform_condition=float(np.linalg.cond(Ps)),response_transform_condition=float(np.linalg.cond(Pr))))
    return rows,windows

def main():
    p=argparse.ArgumentParser();p.add_argument('--charge-ratio',type=float,required=True);p.add_argument('--frequencies',nargs='+',type=float,default=[.002,.004,.006]);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    rows,wins=solve_ratio(a.charge_ratio,a.frequencies);a.output_dir.mkdir(parents=True,exist_ok=True)
    for fn,data in [('response.csv',rows),('windows.csv',wins)]:
        with (a.output_dir/fn).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
