#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master


def background_derivatives(model: master.Model,radius):
    r=np.asarray(radius,float);M=model.mass;q=model.charge
    if model.name=='bardeen':
        d=r*r+q*q
        m=M*r**3*d**-1.5
        mp=3*M*q*q*r*r*d**-2.5
        mpp=3*M*q*q*(2*r*d**-2.5-5*r**3*d**-3.5)
        L=3*M*q*q*d**-2.5
        Lp=-15*M*q*q*r*d**-3.5
        Lpp=-15*M*q*q*(d**-3.5-7*r*r*d**-4.5)
        LF=7.5*M*r**6*d**-3.5
        LFp=LF*(6/r-7*r/d)
    elif model.name=='hayward':
        d=r**3+2*M*q*q
        m=M*r**3/d
        mp=6*M*M*q*q*r*r/d**2
        mpp=12*M*M*q*q*r/d**2-36*M*M*q*q*r**4/d**3
        L=6*M*M*q*q/d**2
        Lp=-36*M*M*q*q*r*r/d**3
        Lpp=-72*M*M*q*q*r/d**3+324*M*M*q*q*r**4/d**4
        LF=18*M*M*r**7/d**3
        LFp=LF*(7/r-9*r*r/d)
    elif model.name=='fan_wang':
        d=r+q
        m=M*r**3/d**3
        mp=3*M*q*r*r/d**4
        mpp=6*M*q*r/d**4-12*M*q*r*r/d**5
        L=3*M*q/d**4
        Lp=-12*M*q/d**5
        Lpp=60*M*q/d**6
        LF=6*M*r**5/(q*d**5)
        LFp=LF*(5/r-5/d)
    else: raise ValueError(model.name)
    f=1-2*m/r;fp=-2*mp/r+2*m/r**2
    a=6*m/r-2*r*r*L
    ap=6*mp/r-6*m/r**2-4*r*L-2*r*r*Lp
    app=6*mpp/r-12*mp/r**2+12*m/r**3-4*L-8*r*Lp-2*r*r*Lpp
    return dict(f=f,fp=fp,m=m,L=L,LF=LF,LFp=LFp,a=a,ap=ap,app=app)


def reconstruct_h0(model,omega,radii,values,derivatives):
    radii=np.asarray(radii,float);bg=background_derivatives(model,radii)
    f,fp,LF,LFp,a,ap,app=[bg[x] for x in ('f','fp','LF','LFp','a','ap','app')]
    angular=6.;lam=4.;sl=2.;A=a+lam;den=radii*A;denp=A+radii*ap
    pots=master.potential_matrix(model,'polar',radii,2)
    out=np.zeros((radii.size,values.shape[2]),complex)
    for col in range(values.shape[2]):
        psi=values[:,0,col];phi=values[:,1,col];psip=derivatives[:,0,col];phip=derivatives[:,1,col]
        second=np.empty((len(radii),2),complex)
        for i in range(len(radii)):
            second[i]=-(fp[i]/f[i])*derivatives[i,:,col]-((omega**2*np.eye(2)-f[i]*pots[i])@values[i,:,col])/f[i]**2
        psipp=second[:,0]
        zf=A/sl;zfp=ap/sl;zfpp=app/sl
        z=zf*psi;zp=zfp*psi+zf*psip;zpp=zfpp*psi+2*zfp*psip+zf*psipp
        sqrtLF=np.sqrt(LF)
        varphi=phi/(2*sqrtLF)+model.charge*psi/(radii*sl)
        varphip=phip/(2*sqrtLF)-phi*LFp/(4*LF*sqrtLF)+model.charge*(psip/radii-psi/radii**2)/sl
        num=8*model.charge*f*LF*varphi/radii-2*radii*f*zp-angular*z
        nump=8*model.charge*((fp*LF+f*LFp)*varphi/radii+f*LF*varphip/radii-f*LF*varphi/radii**2)-2*((f+radii*fp)*zp+radii*f*zpp)-angular*zp
        kp=(nump*den-num*denp)/den**2
        out[:,col]=-zp-radii*kp+4*LF*model.charge*varphi/radii**2
    return out
