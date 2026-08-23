#!/usr/bin/env python3

from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
from rbh_polar_nearzone_basis import build_modes,evaluate
from rbh_polar_metric_reconstruction import reconstruct_h0,background_derivatives


def direct_static_fit(model):
    x=model.charge/model.mass
    if model.name=='hayward':
        return (12.806535897920313*x**2-0.508024965615953*x**4
                -0.05285559611179657*x**6-0.07816243496995176*x**8)
    return (215.47136484956638*x**3+95.2369896663257*x**4
            +1.4652862135271463*x**5-48.49048508096146*x**6)


def horizon_basis(model,omega,radius,horizon_offset=2e-6,rtol=5e-10,atol=5e-12):
    rh=master.outer_horizon(model);start=rh+horizon_offset;fp_h=float(model.fp(rh));Vh=master.potential_matrix(model,'polar',rh,2)
    first=Vh/(fp_h-2j*omega);U0=np.eye(2,dtype=complex)+horizon_offset*first;Up0=-1j*omega*U0/float(model.f(start))+first
    y0=np.concatenate([U0.ravel(),Up0.ravel()])
    def rhs(r,y):
        U=y[:4].reshape(2,2);Up=y[4:].reshape(2,2);f=float(model.f(r));fp=float(model.fp(r));V=master.potential_matrix(model,'polar',r,2)
        return np.concatenate([Up.ravel(),(-(fp/f)*Up-((omega**2*np.eye(2)-f*V)@U)/f**2).ravel()])
    sol=solve_ivp(rhs,(start,radius),y0,method='DOP853',rtol=rtol,atol=atol,max_step=min(.25,.10/max(abs(omega),1e-6)))
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y[:,-1];return y[:4].reshape(2,2),y[4:].reshape(2,2)


def _constant_fit(radii,values):
    r=np.asarray(radii,float);X=np.column_stack([np.ones_like(r),1/r,1/r**2]);scale=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/scale,np.asarray(values).real,rcond=1e-13)[0]/scale
    ci=np.linalg.lstsq(X/scale,np.asarray(values).imag,rcond=1e-13)[0]/scale
    return complex(cr[0],ci[0])


def physical_transform(model,modes):
    rr=np.array([60.,80.,100.,120.,160.]);vals=np.zeros((len(rr),2,4),complex);ders=np.zeros_like(vals)
    for j,(e,c) in enumerate(modes):
        v,d=evaluate(c,e,0.,rr);vals[:,:,j]=v;ders[:,:,j]=d
    h0=reconstruct_h0(model,0.,rr,vals,ders);bg=background_derivatives(model,rr);metric=-bg['f'][:,None]*h0
    sqrtLF=np.sqrt(bg['LF']);varphi=vals[:,1,:]/(2*sqrtLF[:,None])+model.charge*vals[:,0,:]/(2*rr[:,None])
    exps=[e for e,_ in modes]
    if model.name=='hayward':
        gs=_constant_fit(rr,metric[:,0]/rr**2);gr=_constant_fit(rr,metric[:,2]*rr**3)
        pes=-exps[1]/2+1;per=-exps[3]/2+1
        es=_constant_fit(rr,varphi[:,1]/rr**pes);er=_constant_fit(rr,varphi[:,3]/rr**per)
        Ps=np.array([[gs,0],[0,es]],complex);Pr=np.array([[gr,0],[0,er]],complex)
    else:
        Ps=np.zeros((2,2),complex);Pr=np.zeros((2,2),complex)
        for j in range(2):
            Ps[0,j]=_constant_fit(rr,metric[:,j]/rr**2);Ps[1,j]=_constant_fit(rr,varphi[:,j]/rr**3)
        for j in range(2):
            Pr[0,j]=_constant_fit(rr,metric[:,j+2]*rr**3);Pr[1,j]=_constant_fit(rr,varphi[:,j+2]*rr**2)
    return np.linalg.inv(Ps),np.linalg.inv(Pr),Ps,Pr


def basis_data(modes,transform,radius,omega):
    value=np.zeros((2,4),complex);deriv=np.zeros((2,4),complex)
    for j,(e,c) in enumerate(modes):
        v,d=evaluate(c,e,omega,[radius]);value[:,j]=v[0];deriv[:,j]=d[0]
    Ts,Tr=transform
    value=np.column_stack([value[:,:2]@Ts,value[:,2:]@Tr]);deriv=np.column_stack([deriv[:,:2]@Ts,deriv[:,2:]@Tr])
    return value,deriv


def response_coeff(model,modes,transform,omega,radius,controls):
    hv,hd=horizon_basis(model,omega,radius,**controls);av,ad=basis_data(modes,transform,radius,omega)
    basis=np.vstack([av,ad]);horizon=np.vstack([hv,hd]);cols=np.linalg.norm(basis,axis=0);balanced=basis/cols
    C=np.linalg.solve(balanced,horizon)/cols[:,None]
    weights=np.linalg.solve(C[:2,:],np.array([1.,0.],complex));amp=C@weights
    return amp[2],amp[3],np.linalg.cond(balanced),amp[0],amp[1]


def extrapolate(radii,values,omega,omega_order):
    r=np.asarray(radii,float);y=np.asarray(values,complex);p=2*(omega_order+1)
    X=np.column_stack([np.ones_like(r),(omega*r)**p,1/r**2]);scale=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/scale,y.real,rcond=1e-13)[0]/scale;ci=np.linalg.lstsq(X/scale,y.imag,rcond=1e-13)[0]/scale
    pred=X@cr+1j*(X@ci);return complex(cr[0],ci[0]),float(np.sqrt(np.mean(abs(y-pred)**2))),pred


def extrapolate_static(radii,values):
    r=np.asarray(radii,float);y=np.asarray(values,complex)
    X=np.column_stack([np.ones_like(r),1/r**2,1/r**4]);scale=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/scale,y.real,rcond=1e-13)[0]/scale;ci=np.linalg.lstsq(X/scale,y.imag,rcond=1e-13)[0]/scale
    pred=X@cr+1j*(X@ci);return complex(cr[0],ci[0]),float(np.sqrt(np.mean(abs(y-pred)**2))),pred


def solve(model_name,ratio,frequencies,match_radii,series_order=18,omega_order=2,log_order=5,mass=1.,horizon_offset=2e-6,rtol=5e-10,atol=5e-12):
    q=ratio*master.extremal_charge(model_name,mass);model=master.build_model(model_name,q,mass)
    modes=build_modes(model_name,q,series_order,omega_order,log_order,mass);Ts,Tr,Ps,Pr=physical_transform(model,modes);transform=(Ts,Tr);controls=dict(horizon_offset=horizon_offset,rtol=rtol,atol=atol)
    static={r:response_coeff(model,modes,transform,0.,r,controls)[0] for r in match_radii};kstatic=direct_static_fit(model)
    static_metric,static_metric_rms,_=extrapolate_static(match_radii,[static[r] for r in match_radii])
    rows=[];windows=[]
    for omega in frequencies:
        values=[];responses=[]
        for r in match_radii:
            c,em,cond,gs,es=response_coeff(model,modes,transform,omega,r,controls);dc=c-static[r];values.append(dc);responses.append(c)
            windows.append(dict(model=model_name,ell_over_ell_ext=ratio,omega_M=omega,match_radius=r,response_real=c.real,response_imag=c.imag,delta_real=dc.real,delta_imag=dc.imag,em_response_real=em.real,em_response_imag=em.imag,grav_source_error=abs(gs-1),em_source_abs=abs(es),condition=cond))
        metric_ratio,metric_ratio_rms,_=extrapolate(match_radii,responses,omega,omega_order)
        dc_metric,rms_metric,pred_metric=extrapolate(match_radii,values,omega,omega_order)
        love_factor=1.0/mass**5
        # Since -f H0 = -E r^2 + R r^-3 + ..., the polar Love number is
        # minus the finite metric response/source ratio in this basis.
        dc=-love_factor*dc_metric; rms=love_factor*rms_metric
        spread=float(love_factor*max(abs(np.asarray(values)-np.asarray(pred_metric))))
        rows.append(dict(model=model_name,ell_over_ell_ext=ratio,ell=q,omega_M=omega,k_static=kstatic,metric_ratio_real=metric_ratio.real,metric_ratio_imag=metric_ratio.imag,metric_ratio_static_real=static_metric.real,metric_ratio_static_imag=static_metric.imag,metric_ratio_fit_rms=metric_ratio_rms,metric_ratio_static_fit_rms=static_metric_rms,raw_metric_delta_real=dc_metric.real,raw_metric_delta_imag=dc_metric.imag,raw_delta_real=dc.real,raw_delta_imag=dc.imag,k_absolute_real=kstatic+dc.real,k_absolute_imag=dc.imag,window_fit_rms=rms,window_max_residual=spread,maximum_omega_r=max(match_radii)*abs(omega),source_transform_condition=float(np.linalg.cond(Ps)),response_transform_condition=float(np.linalg.cond(Pr))))
    return rows,windows


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['hayward','fan_wang'],required=True);p.add_argument('--charge-ratio',type=float,required=True);p.add_argument('--frequencies',nargs='+',type=float,default=[.002,.004,.006]);p.add_argument('--match-radii',nargs='+',type=float,default=[10,12,14,16]);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    rows,windows=solve(args.model,args.charge_ratio,args.frequencies,args.match_radii)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for name,data in [('response.csv',rows),('windows.csv',windows)]:
        with (args.output_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(rows,indent=2))
if __name__=='__main__': main()
