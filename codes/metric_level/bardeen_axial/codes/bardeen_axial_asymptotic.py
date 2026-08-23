#!/usr/bin/env python3

from __future__ import annotations
from functools import lru_cache
import numpy as np
import sympy as sp

def _symbolic():
    t=sp.symbols('t',positive=True);M,q=sp.symbols('M q',positive=True);r=t**-2
    D=r**2+q**2;m=M*r**3*D**(-sp.Rational(3,2));L=3*M*q**2*D**(-sp.Rational(5,2));LF=sp.Rational(15,2)*M*r**6*D**(-sp.Rational(7,2));f=1-2*m/r
    dr=lambda z:-sp.Rational(1,2)*t**3*sp.diff(z,t);fp=dr(f);lp=dr(sp.log(LF));lpp=dr(lp)
    dm=-sp.Rational(1,2)*fp*lp+f*(-sp.Rational(1,2)*lpp+sp.Rational(1,4)*lp**2)
    v11=6/r**2-6*m/r**3+2*L;v12=-sp.sqrt(16*LF)*q/r**3;v22=6/r**2+dm+4*q**2*LF/r**4
    return t,M,q,(f,fp,v11,v12,v22)
_T,_M,_Q,_EXPRS=_symbolic();_MAX=44

def _symbolic_coeffs(expr):
    s=sp.series(expr.subs(_M,1),_T,0,_MAX+1).removeO().expand();out=[sp.Integer(0)]*(_MAX+1)
    for term in sp.Add.make_args(s):
        c,e=term.as_coeff_exponent(_T);p=int(e)
        if 0<=p<=_MAX:out[p]=sp.simplify(c)
    return out
_COEFFS=[_symbolic_coeffs(e) for e in _EXPRS]
_FUNCS=[[sp.lambdify(_Q,c,'numpy') if c!=0 else None for c in arr] for arr in _COEFFS]

@lru_cache(maxsize=128)
def series(charge:float,order:int,mass:float=1.0):
    maxpow=order+4
    if maxpow>_MAX:raise ValueError(f'order too large; maximum power {_MAX}')
    vals=[]
    q=float(charge);M=float(mass)
    
    if abs(M-1.)>1e-14:raise NotImplementedError('fast series currently normalized to M=1')
    for funcs in _FUNCS:
        a=np.zeros(maxpow+1,complex)
        for p in range(maxpow+1):
            if funcs[p] is not None:a[p]=complex(funcs[p](q))
        vals.append(a)
    f,fp,v11,v12,v22=vals;V=np.zeros((maxpow+1,2,2),complex);V[:,0,0]=v11;V[:,0,1]=v12;V[:,1,0]=v12;V[:,1,1]=v22
    return f,fp,V
