#!/usr/bin/env python3

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import einstein_nled_master as master
import scalar_shell_eft_all_models as scalar


def complex_pair(value):
    return [float(value.real), float(value.imag)]


def relative_spread(values):
    values=np.asarray(values,dtype=complex)
    return float(np.max(np.abs(values-values[len(values)//2]))/max(abs(values[len(values)//2]),1e-30))


def run():
    report={}
    for model in ("bardeen","hayward","fan_wang"):
        q=.70*master.extremal_charge(model,scalar.MASS);omega=5e-4
        radii=np.array([18.,24.,30.])
        bundle=scalar.solution_bundle(model,q,omega,radii)
        schw=scalar.solution_bundle(model,0.,omega,radii)
        direct=[scalar.response_from_bundle(bundle,r)-scalar.response_from_bundle(schw,r) for r in radii]

        reference=[]
        single=np.array([24.])
        full,fd=scalar.outside_profile(model,q,omega,single)
        sfull,sfd=scalar.outside_profile(model,0.,omega,single)
        for rr in (100.,120.,140.):
            src,srcd,resp,respd=scalar.basis_pair_profile(model,q,omega,single,rr)
            c=np.linalg.solve(np.array([[src[0],resp[0]],[srcd[0],respd[0]]],complex),np.array([full[0],fd[0]],complex))
            src,srcd,resp,respd=scalar.basis_pair_profile(model,0.,omega,single,rr)
            cs=np.linalg.solve(np.array([[src[0],resp[0]],[srcd[0],respd[0]]],complex),np.array([sfull[0],sfd[0]],complex))
            reference.append(c[1]/c[0]-cs[1]/cs[0])

        master_r=scalar.SHELL_RADII
        b=scalar.solution_bundle(model,q,omega,master_r); bs=scalar.solution_bundle(model,0.,omega,master_r)
        profile=3/(4*np.pi)*(scalar.renormalized_profile_from_bundle(model,q,b)-scalar.renormalized_profile_from_bundle(model,0.,bs))
        windows={}
        for low in (12.,16.,20.):
            mask=master_r>=low
            fit=scalar.extrapolate_profile_values(master_r[mask],profile[mask])
            windows[f"{low:g}-80M"]={"response":complex_pair(fit['response']),"relative_rms_residual":fit['relative_rms_residual']}
        report[model]={
            "test_point":{"ell_over_ell_ext":.70,"omega_M":omega},
            "direct_match_radius_values":[complex_pair(v) for v in direct],
            "direct_match_radius_relative_spread":relative_spread(direct),
            "basis_reference_radius_values":[complex_pair(v) for v in reference],
            "basis_reference_radius_relative_spread":relative_spread(reference),
            "shell_fit_windows":windows,
        }
    out=Path(__file__).resolve().parent/'results'/'scalar_shell_eft_convergence.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': run()
