#!/usr/bin/env python3



from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
from bardeen_polar_dynamic_frobenius import build_modes, evaluate_dynamic_mode


def direct_static_fit(model):
    x=model.charge/model.mass
    return (0.08022747739312462*x**2+10.76112455362971*x**4
            -0.14943707892554278*x**6-0.1809082206985172*x**8)


def horizon_basis(model,omega,radius,horizon_offset=2e-6,rtol=2e-11,atol=2e-13):
    rh=master.outer_horizon(model); start=rh+horizon_offset
    fp_h=float(model.fp(rh)); Vh=master.potential_matrix(model,'polar',rh,2)
    first=Vh/(fp_h-2j*omega)
    U0=np.eye(2,dtype=complex)+horizon_offset*first
    Up0=-1j*omega*U0/float(model.f(start))+first
    y0=np.concatenate([U0.ravel(),Up0.ravel()])
    def rhs(r,y):
        U=y[:4].reshape(2,2);Up=y[4:].reshape(2,2)
        f=float(model.f(r));fp=float(model.fp(r));V=master.potential_matrix(model,'polar',r,2)
        Upp=-(fp/f)*Up-((omega**2*np.eye(2)-f*V)@U)/f**2
        return np.concatenate([Up.ravel(),Upp.ravel()])
    sol=solve_ivp(rhs,(start,radius),y0,method='DOP853',rtol=rtol,atol=atol,
                  max_step=min(0.25,0.10/max(omega,1e-6)))
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y[:,-1]
    return y[:4].reshape(2,2),y[4:].reshape(2,2)


def build_static_modes(model,order,omega_order=2,log_order=4):
    modes=build_modes(model.charge,order,omega_order,log_order)
    metric_scales=np.array([6.0,1.0,1.0,1.0])
    return modes,metric_scales


def static_basis_data(modes,metric_scales,radius,omega):
    value=np.zeros((2,4),complex);derivative=np.zeros((2,4),complex)
    for j,(e,c) in enumerate(modes):
        v,d=evaluate_dynamic_mode(c,e,omega,np.array([radius]))
        value[:,j]=v[0]/metric_scales[j]
        derivative[:,j]=d[0]/metric_scales[j]
    return value,derivative


def response_coefficient(model,modes,scales,omega,radius,controls):
    hv,hd=horizon_basis(model,omega,radius,**controls)
    av,ad=static_basis_data(modes,scales,radius,omega)
    basis=np.vstack([av,ad])
    horizon=np.vstack([hv,hd])
    
    
    column_scales=np.linalg.norm(basis,axis=0)
    if np.any(column_scales==0):
        raise RuntimeError('Degenerate near-zone basis column')
    balanced=basis/column_scales
    X=np.linalg.solve(balanced,horizon)
    C=X/column_scales[:,None]
    
    weights=np.linalg.solve(C[:2,:],np.array([1.0,0.0],complex))
    amplitudes=C@weights
    return amplitudes[2],amplitudes[3],np.linalg.cond(balanced)


def extrapolate_windows(radii,values,omega,omega_order):
    r=np.asarray(radii,float); y=np.asarray(values,complex)
    
    
    
    omitted_power=2*(omega_order+1)
    X=np.column_stack([np.ones_like(r),(omega*r)**omitted_power,1/r**2])
    scale=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/scale,y.real,rcond=1e-13)[0]/scale
    ci=np.linalg.lstsq(X/scale,y.imag,rcond=1e-13)[0]/scale
    pred=X@cr+1j*(X@ci)
    rms=float(np.sqrt(np.mean(np.abs(y-pred)**2)))
    return complex(cr[0],ci[0]),rms,pred


def extrapolate_static(radii,values):
    r=np.asarray(radii,float);y=np.asarray(values,complex)
    X=np.column_stack([np.ones_like(r),1/r**2,1/r**4]);scale=np.linalg.norm(X,axis=0)
    cr=np.linalg.lstsq(X/scale,y.real,rcond=1e-13)[0]/scale
    ci=np.linalg.lstsq(X/scale,y.imag,rcond=1e-13)[0]/scale
    pred=X@cr+1j*(X@ci)
    return complex(cr[0],ci[0]),float(np.sqrt(np.mean(np.abs(y-pred)**2))),pred


def run(args):
    qext=master.extremal_charge('bardeen',args.mass)
    q=args.charge_ratio*qext
    model=master.build_model('bardeen',q,args.mass)
    modes,scales=build_static_modes(model,args.series_order,args.omega_order,args.log_order)
    controls=dict(horizon_offset=args.horizon_offset,rtol=args.rtol,atol=args.atol)
    static_by_r={}
    for r in args.match_radii:
        static_by_r[r]=response_coefficient(model,modes,scales,0.0,r,controls)[0]
    static_metric,static_metric_rms,_=extrapolate_static(args.match_radii,[static_by_r[r] for r in args.match_radii])
    k_static=direct_static_fit(model)
    rows=[]; window_rows=[]
    for omega in args.frequencies:
        corrections=[];responses=[]
        for r in args.match_radii:
            c,em,cond=response_coefficient(model,modes,scales,omega,r,controls)
            dc=c-static_by_r[r]
            corrections.append(dc);responses.append(c)
            window_rows.append(dict(omega_M=omega*args.mass,match_radius_over_M=r/args.mass,
                                    response_real=c.real,response_imag=c.imag,
                                    static_response_real=static_by_r[r].real,
                                    correction_real=dc.real,correction_imag=dc.imag,
                                    em_response_real=em.real,em_response_imag=em.imag,
                                    basis_condition_number=cond))
        metric_ratio,metric_ratio_rms,_=extrapolate_windows(args.match_radii,responses,omega,args.omega_order)
        dc0_metric,rms_metric,pred_metric=extrapolate_windows(args.match_radii,corrections,omega,args.omega_order)
        love_factor=1.0/args.mass**5
        # Since -f H0 = -E r^2 + R r^-3 + ..., the polar Love number is
        # minus the finite metric response/source ratio in this basis.
        dc0=-love_factor*dc0_metric
        rms=love_factor*rms_metric
        pred=love_factor*np.asarray(pred_metric)
        kval=k_static+dc0
        spread=love_factor*max(abs(np.asarray(corrections)-np.asarray(pred_metric)))
        rows.append(dict(omega_M=omega*args.mass,k_static_reference=k_static,
                         metric_ratio_real=metric_ratio.real,metric_ratio_imag=metric_ratio.imag,
                         metric_ratio_static_real=static_metric.real,metric_ratio_static_imag=static_metric.imag,
                         metric_ratio_fit_rms=metric_ratio_rms,metric_ratio_static_fit_rms=static_metric_rms,
                         delta_k_real=dc0.real,delta_k_imag=dc0.imag,
                         k_dynamic_real=kval.real,k_dynamic_imag=kval.imag,
                         window_fit_rms=rms,window_max_residual=float(spread),
                         maximum_omega_r=max(args.match_radii)*omega))
    out=args.output_dir; (out/'results').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    with (out/'results'/'bardeen_polar_dynamic_tln.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'results'/'bardeen_polar_window_data.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(window_rows[0]));w.writeheader();w.writerows(window_rows)
    om=np.array([x['omega_M'] for x in rows]); kr=np.array([x['k_dynamic_real'] for x in rows]);ki=np.array([x['k_dynamic_imag'] for x in rows])
    fig,ax=plt.subplots(figsize=(6.8,4.6));ax.plot(om,kr,'o-',label=r'$\mathrm{Re}\,k_{20}^{\rm polar}(\omega)$');ax.axhline(k_static,ls='--',label='Direct static result');ax.set_xlabel(r'$M\omega$');ax.set_ylabel(r'$k_{20}^{\rm polar}$');ax.set_title(rf'Bardeen polar near-zone response: $q/q_{{\rm ext}}={args.charge_ratio:g}$');ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout();fig.savefig(out/'figures'/'bardeen_polar_dynamic_tln.png',dpi=220);fig.savefig(out/'figures'/'bardeen_polar_dynamic_tln.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.8,4.6));ax.plot(om,ki,'o-');ax.set_xlabel(r'$M\omega$');ax.set_ylabel(r'$\mathrm{Im}\,k_{20}^{\rm polar}(\omega)$');ax.set_title('Bardeen polar dissipative response');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'figures'/'bardeen_polar_dynamic_tln_imag.png',dpi=220);fig.savefig(out/'figures'/'bardeen_polar_dynamic_tln_imag.pdf');plt.close(fig)
    summary=dict(model='bardeen',parity='polar',mass=args.mass,charge=q,charge_ratio=args.charge_ratio,
                 definition='Near-zone source/response coefficient of reconstructed -f H0, with the zero-frequency constant fixed by the direct coupled static calculation',
                 static_reference=k_static,match_radii=args.match_radii,frequencies=args.frequencies,
                 series_order=args.series_order,omega_order=args.omega_order,log_order=args.log_order,
                 calibration='K(omega)=k_static-[c_metric(omega)-c_metric(0)]/M^5',
                 static_fit_formula='k_static=0.0802275 (q/M)^2+10.7611 (q/M)^4-0.149437 (q/M)^6-0.180908 (q/M)^8',
                 precision_note='At q/q_ext=0.5, increasing the direct static radial order from 22 to 24 changes k_static by 1.3e-5 fractionally.',
                 important_note='The finite-frequency correction is extracted from the full coupled ingoing solution in the reconstructed metric near-zone basis. No frequency-dependent normalization and no far-zone scattering subtraction are used.')
    (out/'results'/'bardeen_polar_dynamic_tln_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return rows


def parse():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mass',type=float,default=1.0);p.add_argument('--charge-ratio',type=float,default=.5);p.add_argument('--series-order',type=int,default=18);p.add_argument('--omega-order',type=int,default=2);p.add_argument('--log-order',type=int,default=4);p.add_argument('--frequencies',type=float,nargs='+',default=[.002,.004,.006,.008,.010]);p.add_argument('--match-radii',type=float,nargs='+',default=[18.,22.,26.,30.]);p.add_argument('--horizon-offset',type=float,default=2e-6);p.add_argument('--rtol',type=float,default=2e-11);p.add_argument('--atol',type=float,default=2e-13);p.add_argument('--output-dir',type=Path,default=HERE.parent);return p.parse_args()
if __name__=='__main__':
 rows=run(parse());
 for x in rows: print(f"Momega={x['omega_M']:.4f}  k={x['k_dynamic_real']:.9g}{x['k_dynamic_imag']:+.3g}i  max(omega r)={x['maximum_omega_r']:.3f}")
