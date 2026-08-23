#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
from rbh_polar_asymptotic import asymptotic_series


def conv(a,b,nmax):
    out=np.zeros(nmax+1,dtype=complex)
    for i,x in enumerate(a):
        if i>nmax: break
        for j,y in enumerate(b):
            if i+j>nmax: break
            out[i+j]+=x*y
    return out


def scalar_matrix_conv(a,b,nmax):
    out=np.zeros((nmax+1,)+b.shape[1:],dtype=complex)
    for i,x in enumerate(a):
        if i>nmax: break
        for j in range(min(len(b),nmax-i+1)): out[i+j]+=x*b[j]
    return out


def operator_series(model_name,charge,order,mass=1.0):
    f,fp,V=asymptotic_series(model_name,float(charge),order+4,float(mass))
    nmax=min(len(f),len(fp),len(V))-1
    return conv(f,f,nmax),conv(f,fp,nmax),scalar_matrix_conv(f,V,nmax),V


def mode_roots(model_name,charge,order=30,mass=1.0):
    _,_,_,V=operator_series(model_name,charge,order,mass)
    leading=np.real_if_close(V[4]).real
    vals,vecs=np.linalg.eigh(leading)
    if abs(vals[1]-vals[0])<1e-8:
        vals=np.array([float(np.mean(vals)),float(np.mean(vals))]);vecs=np.eye(2)
    roots=[]
    for i,val in enumerate(vals):
        nu=(-1+np.sqrt(max(1+4*val,0)))/2
        roots.extend([(-2*(nu+1),vecs[:,i]),(2*nu,vecs[:,i])])
    return roots,vals


def dynamic_mode(model_name,charge,exponent,vector,radial_order=18,omega_order=2,log_order=5,mass=1.0):
    f2,ffp,fV,_=operator_series(model_name,charge,radial_order+4*omega_order+4,mass)
    c=np.zeros((omega_order+1,radial_order+1,log_order+1,2),complex)
    c[0,0,0]=np.asarray(vector,complex);V4=fV[4]
    for k in range(omega_order+1):
        base=exponent-4*k
        for n in range(radial_order+1):
            if k==0 and n==0: continue
            e=base+n;known=np.zeros((log_order+1,2),complex)
            for p,coef in enumerate(f2):
                mm=n-p
                if mm<0 or mm>radial_order or (p==0 and mm==n): continue
                em=base+mm
                for j in range(log_order+1):
                    known[j]+=coef*.25*em*(em+2)*c[k,mm,j]
                    if j+1<=log_order: known[j]+=coef*.5*(em+1)*(j+1)*c[k,mm,j+1]
                    if j+2<=log_order: known[j]+=coef*.25*(j+2)*(j+1)*c[k,mm,j+2]
            for p,coef in enumerate(ffp):
                mm=n+2-p
                if mm<0 or mm>radial_order: continue
                em=base+mm
                for j in range(log_order+1):
                    known[j]+=coef*(-.5*em)*c[k,mm,j]
                    if j+1<=log_order: known[j]+=coef*(-.5*(j+1))*c[k,mm,j+1]
            for p,mat in enumerate(fV):
                mm=n+4-p
                if mm<0 or mm>radial_order or (p==4 and mm==n): continue
                for j in range(log_order+1): known[j]-=mat@c[k,mm,j]
            if k>0: known+=c[k-1,n]
            A=.25*e*(e+2)*np.eye(2)-V4;Ap=.5*(e+1)
            block=np.zeros(((log_order+1)*2,(log_order+1)*2),complex);rhs=-known.reshape(-1)
            for j in range(log_order+1):
                block[2*j:2*j+2,2*j:2*j+2]=A
                if j+1<=log_order: block[2*j:2*j+2,2*(j+1):2*(j+2)]=Ap*(j+1)*np.eye(2)
                if j+2<=log_order: block[2*j:2*j+2,2*(j+2):2*(j+3)]=.25*(j+2)*(j+1)*np.eye(2)
            u,s,vh=np.linalg.svd(A);tol=1e-11*max(1.,s[0]);null=vh.conj().T[:,s<tol]
            if null.shape[1]:
                constraints=np.zeros((null.shape[1],block.shape[1]),complex);constraints[:,:2]=null.conj().T
                sol=np.linalg.lstsq(np.vstack([block,constraints]),np.concatenate([rhs,np.zeros(null.shape[1])]),rcond=1e-12)[0]
                err=np.linalg.norm(block@sol-rhs)/(1+np.linalg.norm(rhs))
                if err>5e-8: raise RuntimeError(f'log order insufficient model={model_name} k={k} n={n} e={e}: {err}')
            else: sol=np.linalg.solve(block,rhs)
            c[k,n]=sol.reshape(log_order+1,2)
    return c


def evaluate(coefficients,exponent,omega,radii):
    radii=np.atleast_1d(np.asarray(radii,float));K,N,L,_=coefficients.shape
    values=np.zeros((len(radii),2),complex);derivs=np.zeros_like(values);w=omega**2
    for ir,r in enumerate(radii):
        t=r**-.5;logt=np.log(t)
        for k in range(K):
            base=exponent-4*k;wk=w**k
            for n in range(N):
                e=base+n;poly=np.zeros(2,complex);dpoly=np.zeros(2,complex)
                for j in range(L):
                    poly+=coefficients[k,n,j]*logt**j
                    if j: dpoly+=j*coefficients[k,n,j]*logt**(j-1)
                values[ir]+=wk*t**e*poly
                derivs[ir]+=-.5*wk*t**(e+2)*(e*poly+dpoly)
    return values,derivs


def build_modes(model_name,charge,radial_order=18,omega_order=2,log_order=5,mass=1.0):
    roots,_=mode_roots(model_name,charge,radial_order+4*omega_order+4,mass)
    source_candidates=sorted(roots,key=lambda x:x[0])[:2]
    response_candidates=sorted(roots,key=lambda x:x[0])[-2:]
    gsrc=max(source_candidates,key=lambda x:abs(x[1][0]));esrc=max(source_candidates,key=lambda x:abs(x[1][1]))
    gresp=max(response_candidates,key=lambda x:abs(x[1][0]));eresp=max(response_candidates,key=lambda x:abs(x[1][1]))
    return [(e,dynamic_mode(model_name,charge,e,v,radial_order,omega_order,log_order,mass)) for e,v in (gsrc,esrc,gresp,eresp)]
