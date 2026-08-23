#!/usr/bin/env python3

from __future__ import annotations
import numpy as np
from bardeen_axial_asymptotic import series

def conv(a,b,n):
    out=np.zeros(n+1,complex)
    for i,x in enumerate(a):
        if i>n:break
        for j,y in enumerate(b):
            if i+j>n:break
            out[i+j]+=x*y
    return out

def smconv(a,b,n):
    out=np.zeros((n+1,)+b.shape[1:],complex)
    for i,x in enumerate(a):
        if i>n:break
        for j in range(min(len(b),n-i+1)):out[i+j]+=x*b[j]
    return out

def operator_series(q,order,mass=1.):
    f,fp,V=series(float(q),order+4,float(mass));n=min(len(f),len(fp),len(V))-1
    return conv(f,f,n),conv(f,fp,n),smconv(f,V,n),V

def mode_roots(q,order=30,mass=1.):
    *_,V=operator_series(q,order,mass)
    lead=np.real_if_close(V[4]).real
    vals,vecs=np.linalg.eigh(lead);roots=[]
    for i,val in enumerate(vals):
        nu=(-1+np.sqrt(max(1+4*val,0)))/2
        roots.extend([(-2*(nu+1),vecs[:,i]),(2*nu,vecs[:,i])])
    return roots,vals

def dynamic_mode(q,exponent,vector,radial_order=22,omega_order=2,log_order=5,mass=1.):
    f2,ffp,fV,_=operator_series(q,radial_order+4*omega_order+4,mass)
    c=np.zeros((omega_order+1,radial_order+1,log_order+1,2),complex);c[0,0,0]=vector
    V4=fV[4]
    for k in range(omega_order+1):
        base=exponent-4*k
        for n in range(radial_order+1):
            if k==0 and n==0:continue
            e=base+n;known=np.zeros((log_order+1,2),complex)
            for p,coef in enumerate(f2):
                mm=n-p
                if mm<0 or mm>radial_order or (p==0 and mm==n):continue
                em=base+mm
                for j in range(log_order+1):
                    known[j]+=coef*.25*em*(em+2)*c[k,mm,j]
                    if j+1<=log_order:known[j]+=coef*.5*(em+1)*(j+1)*c[k,mm,j+1]
                    if j+2<=log_order:known[j]+=coef*.25*(j+2)*(j+1)*c[k,mm,j+2]
            for p,coef in enumerate(ffp):
                mm=n+2-p
                if mm<0 or mm>radial_order:continue
                em=base+mm
                for j in range(log_order+1):
                    known[j]+=-.5*coef*em*c[k,mm,j]
                    if j+1<=log_order:known[j]+=-.5*coef*(j+1)*c[k,mm,j+1]
            for p,mat in enumerate(fV):
                mm=n+4-p
                if mm<0 or mm>radial_order or (p==4 and mm==n):continue
                for j in range(log_order+1):known[j]-=mat@c[k,mm,j]
            if k>0:known+=c[k-1,n]
            A=.25*e*(e+2)*np.eye(2)-V4;Ap=.5*(e+1)
            block=np.zeros(((log_order+1)*2,(log_order+1)*2),complex);rhs=-known.reshape(-1)
            for j in range(log_order+1):
                block[2*j:2*j+2,2*j:2*j+2]=A
                if j+1<=log_order:block[2*j:2*j+2,2*(j+1):2*(j+2)]=Ap*(j+1)*np.eye(2)
                if j+2<=log_order:block[2*j:2*j+2,2*(j+2):2*(j+3)]=.25*(j+2)*(j+1)*np.eye(2)
            u,s,vh=np.linalg.svd(A);tol=1e-11*max(1.,s[0]);null=vh.conj().T[:,s<tol]
            if null.shape[1]:
                constraints=np.zeros((null.shape[1],block.shape[1]),complex);constraints[:,:2]=null.conj().T
                sol=np.linalg.lstsq(np.vstack([block,constraints]),np.concatenate([rhs,np.zeros(null.shape[1])]),rcond=1e-12)[0]
                err=np.linalg.norm(block@sol-rhs)/(1+np.linalg.norm(rhs))
                if err>5e-8:raise RuntimeError(f'log order insufficient k={k} n={n} e={e}: {err}')
            else:sol=np.linalg.solve(block,rhs)
            c[k,n]=sol.reshape(log_order+1,2)
    return c

def evaluate(c,e,omega,radii):
    rr=np.atleast_1d(np.asarray(radii,float));K,N,L,_=c.shape
    val=np.zeros((len(rr),2),complex);der=np.zeros_like(val);w=omega**2
    for ir,r in enumerate(rr):
        t=r**-.5;lt=np.log(t)
        for k in range(K):
            base=e-4*k;wk=w**k
            for n in range(N):
                ee=base+n;poly=np.zeros(2,complex);dpoly=np.zeros(2,complex)
                for j in range(L):
                    poly+=c[k,n,j]*lt**j
                    if j:dpoly+=j*c[k,n,j]*lt**(j-1)
                val[ir]+=wk*t**ee*poly
                der[ir]+=-.5*wk*t**(ee+2)*(ee*poly+dpoly)
    return val,der

def build_modes(q,radial_order=22,omega_order=2,log_order=5,mass=1.):
    roots,_=mode_roots(q,radial_order+4*omega_order+4,mass)
    gsrc=min(roots,key=lambda x:abs(x[0]+6));gresp=min(roots,key=lambda x:abs(x[0]-4))
    rem=[x for x in roots if abs(x[0]-gsrc[0])>1e-8 and abs(x[0]-gresp[0])>1e-8]
    esrc=min(rem,key=lambda x:x[0]);eresp=max(rem,key=lambda x:x[0])
    return [(e,dynamic_mode(q,e,v,radial_order,omega_order,log_order,mass)) for e,v in (gsrc,esrc,gresp,eresp)]
