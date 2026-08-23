#!/usr/bin/env python3

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0,str(HERE))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
import bardeen_polar_metric_reconstruction as reconstruction
from bardeen_polar_dynamic_frobenius import evaluate_dynamic_mode
from bardeen_polar_nearzone_dynamic_tln import build_static_modes, horizon_basis, static_basis_data


def horizon_trajectory(model,omega,radii,weights,horizon_offset,rtol,atol):
    rh=master.outer_horizon(model); start=rh+horizon_offset
    fp_h=float(model.fp(rh)); Vh=master.potential_matrix(model,'polar',rh,2)
    first=Vh/(fp_h-2j*omega)
    U0=np.eye(2,dtype=complex)+horizon_offset*first
    Up0=-1j*omega*U0/float(model.f(start))+first
    value=U0@weights; derivative=Up0@weights
    y0=np.concatenate([value,derivative])
    def rhs(r,y):
        U=y[:2];Up=y[2:]
        f=float(model.f(r));fp=float(model.fp(r));V=master.potential_matrix(model,'polar',r,2)
        Upp=-(fp/f)*Up-((omega**2*np.eye(2)-f*V)@U)/f**2
        return np.concatenate([Up,Upp])
    sol=solve_ivp(rhs,(start,float(max(radii))),y0,t_eval=np.asarray(radii),method='DOP853',
                  rtol=rtol,atol=atol,max_step=min(.25,.10/max(omega,1e-6)))
    if not sol.success:
        raise RuntimeError(sol.message)
    values=sol.y[:2].T[:,:,None];derivatives=sol.y[2:].T[:,:,None]
    return values,derivatives


def run(args):
    charge=args.charge_ratio*master.extremal_charge('bardeen',args.mass)
    model=master.build_model('bardeen',charge,args.mass)
    modes,scales=build_static_modes(model,args.series_order,args.omega_order,args.log_order)
    hv,hd=horizon_basis(model,args.omega,args.match_radius,
                        horizon_offset=args.horizon_offset,rtol=args.rtol,atol=args.atol)
    av,ad=static_basis_data(modes,scales,args.match_radius,args.omega)
    basis=np.vstack([av,ad]);horizon=np.vstack([hv,hd])
    norms=np.linalg.norm(basis,axis=0);balanced=basis/norms
    coefficients=np.linalg.solve(balanced,horizon)/norms[:,None]
    weights=np.linalg.solve(coefficients[:2],np.array([1.,0.],complex))
    amplitudes=coefficients@weights

    radii=np.linspace(args.radius_min,args.radius_max,args.points)
    numerical_values,numerical_derivatives=horizon_trajectory(
        model,args.omega,radii,weights,args.horizon_offset,args.rtol,args.atol)
    numerical_h0=reconstruction.reconstruct_h0(
        model,args.omega,radii,numerical_values,numerical_derivatives)[:,0]

    basis_values=np.zeros((len(radii),2,4),complex)
    basis_derivatives=np.zeros_like(basis_values)
    for j,(exponent,series) in enumerate(modes):
        value,derivative=evaluate_dynamic_mode(series,exponent,args.omega,radii)
        basis_values[:,:,j]=value/scales[j]
        basis_derivatives[:,:,j]=derivative/scales[j]
    near_values=np.einsum('rij,j->ri',basis_values,amplitudes)[:,:,None]
    near_derivatives=np.einsum('rij,j->ri',basis_derivatives,amplitudes)[:,:,None]
    near_h0=reconstruction.reconstruct_h0(
        model,args.omega,radii,near_values,near_derivatives)[:,0]

    f=np.asarray(model.f(radii),float)
    numerical_metric=-f*numerical_h0
    near_metric=-f*near_h0
    rel=np.abs(numerical_metric-near_metric)/np.maximum(np.abs(numerical_metric),1e-30)

    out=args.output_dir;(out/'results').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    with (out/'results'/'bardeen_polar_metric_projection.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['radius_over_M','metric_numeric_real','metric_numeric_imag','metric_nearzone_real','metric_nearzone_imag','relative_difference'])
        for r,a,b,e in zip(radii,numerical_metric,near_metric,rel):
            writer.writerow([r,a.real,a.imag,b.real,b.imag,e])
    summary=dict(omega_M=args.omega*args.mass,charge_ratio=args.charge_ratio,
                 match_radius=args.match_radius,
                 gravitational_source_amplitude=[amplitudes[0].real,amplitudes[0].imag],
                 electromagnetic_source_amplitude=[amplitudes[1].real,amplitudes[1].imag],
                 gravitational_response_amplitude=[amplitudes[2].real,amplitudes[2].imag],
                 electromagnetic_response_amplitude=[amplitudes[3].real,amplitudes[3].imag],
                 maximum_relative_metric_difference=float(max(rel)))
    (out/'results'/'bardeen_polar_metric_projection_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

    fig,ax=plt.subplots(figsize=(6.8,4.6))
    ax.plot(radii,(numerical_metric/radii**2).real,label='Full ingoing solution')
    ax.plot(radii,(near_metric/radii**2).real,'--',label='Near-zone source/response expansion')
    ax.set_xlabel(r'$r/M$');ax.set_ylabel(r'$\mathrm{Re}[-fH_0]/r^2$')
    ax.set_title(rf'Bardeen polar metric projection: $M\omega={args.omega:g}$')
    ax.grid(alpha=.25);ax.legend(frameon=False);fig.tight_layout()
    fig.savefig(out/'figures'/'bardeen_polar_metric_projection.png',dpi=220)
    fig.savefig(out/'figures'/'bardeen_polar_metric_projection.pdf');plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.8,4.6))
    ax.semilogy(radii,rel)
    ax.set_xlabel(r'$r/M$');ax.set_ylabel('Relative metric difference')
    ax.set_title('Near-zone reconstruction residual')
    ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(out/'figures'/'bardeen_polar_metric_projection_residual.png',dpi=220)
    fig.savefig(out/'figures'/'bardeen_polar_metric_projection_residual.pdf');plt.close(fig)
    return summary


def parse():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mass',type=float,default=1.)
    p.add_argument('--charge-ratio',type=float,default=.5)
    p.add_argument('--omega',type=float,default=.004)
    p.add_argument('--match-radius',type=float,default=14.)
    p.add_argument('--radius-min',type=float,default=8.)
    p.add_argument('--radius-max',type=float,default=18.)
    p.add_argument('--points',type=int,default=121)
    p.add_argument('--series-order',type=int,default=18)
    p.add_argument('--omega-order',type=int,default=2)
    p.add_argument('--log-order',type=int,default=5)
    p.add_argument('--horizon-offset',type=float,default=2e-6)
    p.add_argument('--rtol',type=float,default=5e-10)
    p.add_argument('--atol',type=float,default=5e-12)
    p.add_argument('--output-dir',type=Path,default=HERE.parent)
    return p.parse_args()

if __name__=='__main__':
    result=run(parse())
    print(json.dumps(result,indent=2))
