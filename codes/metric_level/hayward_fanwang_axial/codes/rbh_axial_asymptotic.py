#!/usr/bin/env python3

from __future__ import annotations
from functools import lru_cache
import numpy as np
import sympy as sp

_T=sp.symbols('t', positive=True)
_M,_Q=sp.symbols('M q', positive=True)
_X=sp.symbols('x', positive=True)

def _expressions(model_name: str):
    x=_X;M=_M;q=_Q
    if model_name=='hayward':
        D=1+2*M*q**2*x**3
        mf=M/D
        L=6*M**2*q**2*x**6/D**2
        LF=18*M**2*x**2/D**3
        sqrtLF=3*sp.sqrt(2)*M*x/D**sp.Rational(3,2)
        first=7*x-9*x/D
        second=-7*x**2-18*x**2/D+27*x**2/D**2
    elif model_name=='fan_wang':
        D=1+q*x
        mf=M/D**3
        L=3*M*q*x**4/D**4
        LF=6*M/(q*D**5)
        sqrtLF=sp.sqrt(6*M/q)/D**sp.Rational(5,2)
        first=5*x-5*x/D
        second=-5*x**2+5*x**2/D**2
    else:
        raise ValueError(model_name)
    f=1-2*mf*x
    
    fp=-x**2*sp.diff(f,x)
    dminus=-sp.Rational(1,2)*fp*first+f*(-sp.Rational(1,2)*second+sp.Rational(1,4)*first**2)
    v11=6*x**2-6*mf*x**3+2*L
    v12=-4*sqrtLF*q*x**3
    v22=6*x**2+dminus+4*q**2*LF*x**4
    return f,fp,v11,v12,v22

_MAX=72
_CACHE={}
for _name in ('hayward','fan_wang'):
    coeff_arrays=[]
    for expr in _expressions(_name):
        ser=sp.series(expr.subs({_M:1,_X:_T**2}),_T,0,_MAX+1).removeO().expand()
        arr=[sp.Integer(0)]*(_MAX+1)
        for term in sp.Add.make_args(ser):
            c,e=term.as_coeff_exponent(_T);p=int(e)
            if 0<=p<=_MAX: arr[p]=sp.simplify(arr[p]+c)
        coeff_arrays.append([sp.lambdify(_Q,c,'numpy') if c!=0 else None for c in arr])
    _CACHE[_name]=coeff_arrays

@lru_cache(maxsize=256)
def asymptotic_series(model_name: str,charge: float,order: int,mass: float=1.0):
    if abs(mass-1.)>1e-14: raise NotImplementedError('series normalized to M=1')
    maxpow=order+4
    if maxpow>_MAX: raise ValueError(f'order too large; maximum power {_MAX}')
    q=float(charge)
    
    if q==0.0:
        f=np.zeros(maxpow+1,complex);fp=np.zeros_like(f);V=np.zeros((maxpow+1,2,2),complex)
        f[0]=1.;f[2]=-2.;fp[4]=2.
        V[4,0,0]=6.;V[6,0,0]=-6.
        V[4,1,1]=6.
        if model_name=='hayward': V[6,1,1]=2.
        return f,fp,V
    vals=[]
    for funcs in _CACHE[model_name]:
        a=np.zeros(maxpow+1,complex)
        for p in range(maxpow+1):
            if funcs[p] is not None: a[p]=complex(funcs[p](q))
        vals.append(a)
    f,fp,v11,v12,v22=vals
    V=np.zeros((maxpow+1,2,2),complex)
    V[:,0,0]=v11;V[:,0,1]=v12;V[:,1,0]=v12;V[:,1,1]=v22
    return f,fp,V
