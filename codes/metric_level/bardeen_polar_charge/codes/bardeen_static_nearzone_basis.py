#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master
import bardeen_polar_jost_response as jost
import bardeen_polar_metric_reconstruction as recon


def conv(a,b,nmax):
    out=np.zeros(nmax+1,dtype=complex)
    for i,x in enumerate(a):
        if i>nmax: break
        for k,y in enumerate(b):
            if i+k>nmax: break
            out[i+k]+=x*y
    return out

def matrix_scalar_conv(a,b,nmax):
    
    out=np.zeros((nmax+1,)+b.shape[1:],dtype=complex)
    for i,x in enumerate(a):
        if i>nmax: break
        for k in range(min(len(b),nmax-i+1)):
            out[i+k]+=x*b[k]
    return out

def static_operator_series(charge, order):
    f,fp,V=jost.bardeen_asymptotic_series(charge,order+4)
    nmax=min(len(f),len(fp),len(V))-1
    f2=conv(f,f,nmax)
    ffp=conv(f,fp,nmax)
    fV=matrix_scalar_conv(f,V,nmax)
    return f2,ffp,fV,V

def frobenius_mode(charge, exponent, vector, order=24, verbose=False):
    f2,ffp,fV,V=static_operator_series(charge,order)
    dim=2
    a=np.zeros((order+1,dim),dtype=complex)
    b=np.zeros_like(a)
    a[0]=np.asarray(vector,dtype=complex)
    V4=fV[4]
    for n in range(1,order+1):
        e=exponent+n
        target=exponent+n+4
        kc=np.zeros(dim,dtype=complex)
        kl=np.zeros(dim,dtype=complex)
        
        for p,coef in enumerate(f2):
            m=n-p
            if m<0 or m>order or (m==n and p==0): continue
            em=exponent+m
            if abs(coef)==0: continue
            kl += coef*(0.25*em*(em+2))*b[m]
            kc += coef*((0.25*em*(em+2))*a[m]+0.5*(em+1)*b[m])
        
        for p,coef in enumerate(ffp):
            m=n+2-p
            if m<0 or m>order: continue
            em=exponent+m
            if abs(coef)==0: continue
            kl += coef*(-0.5*em)*b[m]
            kc += coef*(-0.5*(em*a[m]+b[m]))
        
        for p,mat in enumerate(fV):
            m=n+4-p
            if m<0 or m>order or (m==n and p==4): continue
            if np.max(np.abs(mat))==0: continue
            kl -= mat@b[m]
            kc -= mat@a[m]
        A=0.25*e*(e+2)*np.eye(dim)-V4
        Ap=0.5*(e+1)*np.eye(dim)
        
        u,s,vh=np.linalg.svd(A)
        tol=1e-11*max(1.0,s[0])
        null=vh.conj().T[:,s<tol]
        if null.shape[1]==0:
            b[n]=np.linalg.solve(A,-kl)
            a[n]=np.linalg.solve(A,-kc-Ap@b[n])
        else:
            
            pinv=np.linalg.pinv(A,rcond=1e-12)
            bpart=-pinv@kl
            residual=kl+A@bpart
            if np.linalg.norm(residual)>1e-8*(1+np.linalg.norm(kl)):
                raise RuntimeError(f'log^2 needed at n={n}, residual={residual}')
            
            rhs0=kc+Ap@bpart
            
            G=null.conj().T@Ap@null
            rhsn=-(null.conj().T@rhs0)
            coeff=np.linalg.solve(G,rhsn)
            b[n]=bpart+null@coeff
            rhs=-kc-Ap@b[n]
            apart=pinv@rhs
            
            a[n]=apart-null@(null.conj().T@apart)
            if verbose:
                print('resonance n',n,'e',e,'b',b[n],'solv',np.linalg.norm(A@a[n]-rhs))
    return a,b

def eval_mode(a,b,exponent,r):
    r=np.asarray(r,dtype=float)
    t=r**-0.5
    L=np.log(t)
    vals=[]; ders=[]
    for ti,Li,ri in zip(np.atleast_1d(t),np.atleast_1d(L),np.atleast_1d(r)):
        U=np.zeros(2,dtype=complex); Ur=np.zeros(2,dtype=complex)
        for n in range(len(a)):
            e=exponent+n
            te=ti**e
            U += te*(a[n]+b[n]*Li)
            
            Ur += -0.5*ti**(e+2)*(e*(a[n]+b[n]*Li)+b[n])
        vals.append(U); ders.append(Ur)
    return np.array(vals),np.array(ders)

def mode_roots(charge,order=30):
    _,_,_,V=static_operator_series(charge,order)
    vals,vecs=np.linalg.eigh(np.real(V[4]))
    roots=[]
    for i,val in enumerate(vals):
        nu=(-1+np.sqrt(1+4*val))/2
        roots.append((-2*(nu+1),vecs[:,i]))
        roots.append((2*nu,vecs[:,i]))
    return roots,vals

def integrate_matrix(model,omega,start,stop,value,derivative,rtol=1e-11,atol=1e-13):
    ncol=value.shape[1]
    y0=np.concatenate([value.ravel(),derivative.ravel()])
    def rhs(r,y):
        U=y[:2*ncol].reshape(2,ncol); Up=y[2*ncol:].reshape(2,ncol)
        f=float(model.f(r)); fp=float(model.fp(r)); V=master.potential_matrix(model,'polar',r,2)
        Upp=-(fp/f)*Up-((omega**2*np.eye(2)-f*V)@U)/f**2
        return np.concatenate([Up.ravel(),Upp.ravel()])
    sol=solve_ivp(rhs,(start,stop),y0,method='DOP853',rtol=rtol,atol=atol,max_step=0.3)
    if not sol.success: raise RuntimeError(sol.message)
    y=sol.y[:,-1]
    return y[:2*ncol].reshape(2,ncol),y[2*ncol:].reshape(2,ncol)

def reconstruct_mode_h0(model,radii,vals,ders):
    
    return recon.reconstruct_h0(model,0.0,np.asarray(radii),vals,ders)

def static_extract(charge_ratio=0.5,R=80.0,match=10.0,order=30):
    q=charge_ratio*master.extremal_charge('bardeen',1.0)
    model=master.build_model('bardeen',q,1.0)
    roots,vals=mode_roots(q,order)
    
    roots_sorted=sorted(roots,key=lambda x:x[0])
    print('roots',[(x[0],x[1]) for x in roots_sorted])
    
    candidates=roots
    gsrc=min(candidates,key=lambda x:abs(x[0]+6))
    gresp=min(candidates,key=lambda x:abs(x[0]-4))
    remaining=[x for x in candidates if x is not gsrc and x is not gresp]
    esrc=min(remaining,key=lambda x:x[0]); eresp=max(remaining,key=lambda x:x[0])
    modes=[gsrc,esrc,gresp,eresp]
    coeffs=[]
    for e,v in modes:
        a,b=frobenius_mode(q,e,v,order,verbose=True)
        coeffs.append((e,a,b))
    valsR=np.zeros((2,4),complex); dersR=np.zeros((2,4),complex)
    for j,(e,a,b) in enumerate(coeffs):
        vv,dd=eval_mode(a,b,e,np.array([R])); valsR[:,j]=vv[0]; dersR[:,j]=dd[0]
    valsM,dersM=integrate_matrix(model,0.0,R,match,valsR,dersR)
    
    rh=master.outer_horizon(model); eps=1e-6; start=rh+eps
    Vh=master.potential_matrix(model,'polar',rh,2); fp=float(model.fp(rh))
    first=Vh/fp
    hv=np.eye(2)+eps*first; hd=first
    hvM,hdM=integrate_matrix(model,0.0,start,match,hv,hd)
    asym=np.vstack([valsM,dersM]); hor=np.vstack([hvM,hdM])
    C=np.linalg.solve(asym,hor) 
    S=C[:2,:]
    x=np.linalg.solve(S,np.array([1,0],complex))
    cc=C@x
    print('mode coeff',cc)
    
    rr=np.linspace(R*0.8,R*1.2,41)
    Varr=np.zeros((len(rr),2,4),complex); Darr=np.zeros_like(Varr)
    for j,(e,a,b) in enumerate(coeffs):
        vv,dd=eval_mode(a,b,e,rr); Varr[:,:,j]=vv;Darr[:,:,j]=dd
    h0=reconstruct_mode_h0(model,rr,Varr,Darr)
    
    csrc=np.median(h0[:,0]/rr**2)
    cresp=np.median(h0[:,2]*rr**3)
    print('h0 norms',csrc,cresp)
    ratio=cc[2]*cresp/(cc[0]*csrc)
    print('physical ratio',ratio)
    return ratio,cc,csrc,cresp

if __name__=='__main__':
    static_extract()
