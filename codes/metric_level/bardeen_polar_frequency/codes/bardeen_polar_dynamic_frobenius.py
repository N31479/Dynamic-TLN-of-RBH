#!/usr/bin/env python3

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))
from bardeen_static_nearzone_basis import static_operator_series, mode_roots


def _poly_add(target,source):
    target[:source.shape[0]] += source


def dynamic_frobenius_mode(charge, exponent, vector, radial_order=20, omega_order=2, log_order=4, verbose=False):
    f2,ffp,fV,_=static_operator_series(charge,radial_order+4*omega_order+4)
    dim=2
    c=np.zeros((omega_order+1,radial_order+1,log_order+1,dim),complex)
    c[0,0,0]=np.asarray(vector,complex)
    V4=fV[4]
    for k in range(omega_order+1):
        base=exponent-4*k
        for n in range(radial_order+1):
            if k==0 and n==0: continue
            e=base+n
            known=np.zeros((log_order+1,dim),complex)
            
            for p,coef in enumerate(f2):
                m=n-p
                if m<0 or m>radial_order or (p==0 and m==n): continue
                em=base+m
                for j in range(log_order+1):
                    known[j]+=coef*0.25*em*(em+2)*c[k,m,j]
                    if j+1<=log_order: known[j]+=coef*0.5*(em+1)*(j+1)*c[k,m,j+1]
                    if j+2<=log_order: known[j]+=coef*0.25*(j+2)*(j+1)*c[k,m,j+2]
            
            for p,coef in enumerate(ffp):
                m=n+2-p
                if m<0 or m>radial_order: continue
                em=base+m
                for j in range(log_order+1):
                    known[j]+=coef*(-0.5*em)*c[k,m,j]
                    if j+1<=log_order: known[j]+=coef*(-0.5*(j+1))*c[k,m,j+1]
            
            for p,mat in enumerate(fV):
                m=n+4-p
                if m<0 or m>radial_order or (p==4 and m==n): continue
                for j in range(log_order+1): known[j]-=mat@c[k,m,j]
            
            if k>0:
                known += c[k-1,n]
            A=0.25*e*(e+2)*np.eye(dim)-V4
            Ap=0.5*(e+1)
            
            block=np.zeros(((log_order+1)*dim,(log_order+1)*dim),complex)
            rhs=-known.reshape(-1)
            for j in range(log_order+1):
                block[j*dim:(j+1)*dim,j*dim:(j+1)*dim]=A
                if j+1<=log_order:
                    block[j*dim:(j+1)*dim,(j+1)*dim:(j+2)*dim]=Ap*(j+1)*np.eye(dim)
                if j+2<=log_order:
                    block[j*dim:(j+1)*dim,(j+2)*dim:(j+3)*dim]=0.25*(j+2)*(j+1)*np.eye(dim)
            
            u,s,vh=np.linalg.svd(A);tol=1e-11*max(1.0,s[0]);null=vh.conj().T[:,s<tol]
            if null.shape[1]:
                constraints=np.zeros((null.shape[1],block.shape[1]),complex)
                constraints[:,:dim]=null.conj().T
                block_aug=np.vstack([block,constraints]);rhs_aug=np.concatenate([rhs,np.zeros(null.shape[1])])
                sol,resid,rank,_=np.linalg.lstsq(block_aug,rhs_aug,rcond=1e-12)
                err=np.linalg.norm(block@sol-rhs)/(1+np.linalg.norm(rhs))
                if err>2e-8:
                    raise RuntimeError(f'Insufficient log_order at k={k}, n={n}, e={e}: residual {err:.3e}')
                if verbose: print('resonance',k,n,e,'err',err,'null',null.shape[1])
            else:
                sol=np.linalg.solve(block,rhs)
            c[k,n]=sol.reshape(log_order+1,dim)
    return c


def evaluate_dynamic_mode(coefficients,exponent,omega,radii):
    radii=np.atleast_1d(np.asarray(radii,float));K,N,L,_=coefficients.shape
    values=np.zeros((len(radii),2),complex);derivs=np.zeros_like(values)
    w=omega**2
    for ir,r in enumerate(radii):
        t=r**-.5;logt=np.log(t)
        for k in range(K):
            base=exponent-4*k
            wk=w**k
            for n in range(N):
                e=base+n;poly=np.zeros(2,complex);dpoly=np.zeros(2,complex)
                for j in range(L):
                    poly += coefficients[k,n,j]*logt**j
                    if j>0: dpoly += j*coefficients[k,n,j]*logt**(j-1)
                values[ir]+=wk*t**e*poly
                derivs[ir]+=-0.5*wk*t**(e+2)*(e*poly+dpoly)
    return values,derivs


def build_modes(charge,radial_order=20,omega_order=2,log_order=4):
    roots,_=mode_roots(charge,radial_order+4*omega_order+4)
    gsrc=min(roots,key=lambda x:abs(x[0]+6));gresp=min(roots,key=lambda x:abs(x[0]-4))
    rem=[x for x in roots if abs(x[0]+6)>1e-8 and abs(x[0]-4)>1e-8]
    esrc=min(rem,key=lambda x:x[0]);eresp=max(rem,key=lambda x:x[0])
    out=[]
    for e,v in [gsrc,esrc,gresp,eresp]:
        out.append((e,dynamic_frobenius_mode(charge,e,v,radial_order,omega_order,log_order)))
    return out
