#!/usr/bin/env python3

from __future__ import annotations
from functools import lru_cache
import numpy as np
import mpmath as mp


def eval_components(model_name,x,q,M=1):
    if model_name=='hayward':
        D=1+2*M*q*q*x**3
        mf=M/D;L=6*M*M*q*q*x**6/D**2;LF=18*M*M*x**2/D**3
        sqrtLF=3*mp.sqrt(2)*M*x/D**mp.mpf('1.5')
        f=1-2*mf*x
        
        dmf=-6*M*M*q*q*x**2/D**2
        fp=-x**2*(-2*(dmf*x+mf))
        lp=7*x-9*x/D;lpp=-7*x**2-18*x**2/D+27*x**2/D**2
    elif model_name=='fan_wang':
        D=1+q*x
        mf=M/D**3;L=3*M*q*x**4/D**4;LF=6*M/(q*D**5)
        sqrtLF=mp.sqrt(6*M/q)/D**mp.mpf('2.5')
        f=1-2*mf*x
        dmf=-3*M*q/D**4
        fp=-x**2*(-2*(dmf*x+mf))
        lp=5*x-5*x/D;lpp=-5*x**2+5*x**2/D**2
    else: raise ValueError(model_name)
    kap=1-mp.mpf('.5')*lp/x
    dplus=mp.mpf('.5')*fp*lp+f*(mp.mpf('.5')*lpp+mp.mpf('.25')*lp**2)
    angular=6;lam=4;r=1/x
    a=6*mf*x-2*r*r*L;b=lam+4*LF*q*q*x*x;den=a+lam
    coupling=-2*mp.sqrt(lam)*sqrtLF*q*x**3
    v11=(angular*lam-2*f*lam+a*(a-4*mf*x))*x*x/den+2*f*lam*b*x*x/den**2
    v22=kap*angular*x*x+dplus+4*LF*q*q*(lam+1-f+2*r*r*L+4*f*kap)*x**4/den+8*f*LF*q*q*b*x**4/den**2
    w=(lam+1-f+2*r*r*L+2*f*kap)/den+2*f*b/den**2
    v12=coupling*w
    return (f,fp,v11,v12,v22)


def cauchy(model_name,q,nmax,M=1.0):
    mp.mp.dps=70;N=max(80,5*(nmax+1));rho=mp.mpf('0.07')
    sums=[[mp.mpc(0) for _ in range(5)] for _ in range(nmax+1)]
    for k in range(N):
        theta=2*mp.pi*k/N;z=rho*mp.e**(1j*theta);vals=eval_components(model_name,z,mp.mpf(str(q)),mp.mpf(str(M)))
        phase=mp.mpc(1);base=mp.e**(-1j*theta)
        for n in range(nmax+1):
            for j,val in enumerate(vals): sums[n][j]+=val*phase
            phase*=base
    out=np.zeros((nmax+1,5),complex)
    for n in range(nmax+1):
        scale=N*rho**n
        for j in range(5): out[n,j]=complex(sums[n][j]/scale)
    out[np.abs(out)<1e-30]=0
    return out


@lru_cache(maxsize=128)
def asymptotic_series(model_name: str,charge: float,order: int,mass: float=1.0):
    nmax=(order+4)//2+1;cx=cauchy(model_name,float(charge),nmax,float(mass));max_power=order+4
    f=np.zeros(max_power+1,complex);fp=np.zeros_like(f);V=np.zeros((max_power+1,2,2),complex)
    for n in range(nmax+1):
        p=2*n
        if p<=max_power:
            f[p]=cx[n,0];fp[p]=cx[n,1];V[p,0,0]=cx[n,2];V[p,0,1]=cx[n,3];V[p,1,0]=cx[n,3];V[p,1,1]=cx[n,4]
    return np.real_if_close(f,tol=1000).real.astype(complex),np.real_if_close(fp,tol=1000).real.astype(complex),np.real_if_close(V,tol=1000).real.astype(complex)
