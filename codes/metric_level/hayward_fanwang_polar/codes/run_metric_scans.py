#!/usr/bin/env python3

from __future__ import annotations
import csv,json,math,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE))
from rbh_polar_metric_solver import solve,direct_static_fit
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import einstein_nled_master as master

MODELS=['hayward','fan_wang']
RATIOS=[1e-5,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
CHARGE_FREQS=[.002,.004,.006]
FREQ_RATIOS=[.5,.9,.97]
FREQS=[.0005,.001,.0015,.002,.0025,.003,.0035,.004,.0045,.005,.0055,.006]
MATCH=[10.,12.,14.,16.]


def task(item):
    model,ratio,freqs=item
    return model,ratio,*solve(model,ratio,freqs,MATCH)


def static_curve(model,x):
    q=x*master.extremal_charge(model,1.)
    if model=='hayward':
        return (12.806535897920313*q**2-0.508024965615953*q**4
                -0.05285559611179657*q**6-0.07816243496995176*q**8)
    return (215.47136484956638*q**3+95.2369896663257*q**4
            +1.4652862135271463*q**5-48.49048508096146*q**6)


def run_all():
    tasks=[]
    for model in MODELS:
        tasks.append((model,1e-5,sorted(set(CHARGE_FREQS+FREQS))))
        for ratio in RATIOS[1:]: tasks.append((model,ratio,CHARGE_FREQS))
        for ratio in FREQ_RATIOS:
            if ratio not in RATIOS[1:]: tasks.append((model,ratio,FREQS))
            else: tasks.append((model,ratio,FREQS))
    raw={};windows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(task,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            model,ratio,rows,win=f.result();key=(model,round(ratio,8));raw.setdefault(key,{})
            for row in rows: raw[key][round(row['omega_M'],8)]=row
            windows.extend(win);print(f'metric task {i}/{len(fs)}',flush=True)
    results=ROOT/'results';results.mkdir(exist_ok=True)
    with (results/'metric_window_diagnostics.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(windows[0]));w.writeheader();w.writerows(windows)
    
    charge_rows=[]
    for model in MODELS:
        proxy=raw[(model,round(1e-5,8))]
        for ratio in [0.0]+RATIOS[1:]:
            for om in CHARGE_FREQS:
                row = proxy[round(om,8)] if ratio==0 else raw[(model,round(ratio,8))][round(om,8)]
                kst = 0.0 if ratio==0 else row['k_static']
                # Polar Love normalization is minus the metric response/source ratio.
                dr = -(row['metric_ratio_real']-row['metric_ratio_static_real'])
                di = -(row['metric_ratio_imag']-row['metric_ratio_static_imag'])
                rms=row['window_fit_rms'];res=row['window_max_residual']
                charge_rows.append(dict(model=model,ell_over_ell_ext=ratio,ell=ratio*master.extremal_charge(model,1.),omega_M=om,k_static=kst,metric_ratio_real=row['metric_ratio_real'],metric_ratio_imag=row['metric_ratio_imag'],metric_ratio_static_real=row['metric_ratio_static_real'],metric_ratio_static_imag=row['metric_ratio_static_imag'],delta_real=dr,delta_imag=di,k_dynamic_real=kst+dr,k_dynamic_imag=di,window_fit_rms=rms,window_max_residual=res))
    with (results/'hayward_fanwang_metric_charge_scan.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(charge_rows[0]));w.writeheader();w.writerows(charge_rows)
    
    frequency_rows=[]
    for model in MODELS:
        proxy=raw[(model,round(1e-5,8))]
        for ratio in FREQ_RATIOS:
            kst=direct_static_fit(master.build_model(model,ratio*master.extremal_charge(model,1.)))
            frequency_rows.append(dict(model=model,ell_over_ell_ext=ratio,omega_M=0.,k_static=kst,metric_ratio_real=float('nan'),metric_ratio_imag=0.,metric_ratio_static_real=float('nan'),metric_ratio_static_imag=0.,delta_real=0.,delta_imag=0.,k_dynamic_real=kst,k_dynamic_imag=0.,window_fit_rms=0.,window_max_residual=0.))
            data=raw[(model,round(ratio,8))]
            for om in FREQS:
                row=data[round(om,8)]
                dr=-(row['metric_ratio_real']-row['metric_ratio_static_real']);di=-(row['metric_ratio_imag']-row['metric_ratio_static_imag'])
                frequency_rows.append(dict(model=model,ell_over_ell_ext=ratio,omega_M=om,k_static=kst,metric_ratio_real=row['metric_ratio_real'],metric_ratio_imag=row['metric_ratio_imag'],metric_ratio_static_real=row['metric_ratio_static_real'],metric_ratio_static_imag=row['metric_ratio_static_imag'],delta_real=dr,delta_imag=di,k_dynamic_real=kst+dr,k_dynamic_imag=di,window_fit_rms=row['window_fit_rms'],window_max_residual=row['window_max_residual']))
    with (results/'hayward_fanwang_metric_frequency_scan.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(frequency_rows[0]));w.writeheader();w.writerows(frequency_rows)
    return charge_rows,frequency_rows


def plot(charge_rows,frequency_rows):
    figs=ROOT/'figures';figs.mkdir(exist_ok=True)
    labels={'hayward':'Hayward','fan_wang':'Fan--Wang'}
    for model in MODELS:
        xsm=np.linspace(0,1,500);ks=static_curve(model,xsm)
        fig,ax=plt.subplots(figsize=(7.2,5.0));ax.plot(xsm,ks,'--',lw=1.8,label=r'Static ($M\omega=0$)')
        lines=[]
        for om,mark in zip(CHARGE_FREQS,['o','s','^']):
            sub=sorted([r for r in charge_rows if r['model']==model and abs(r['omega_M']-om)<1e-12],key=lambda r:r['ell_over_ell_ext'])
            x=np.array([r['ell_over_ell_ext'] for r in sub]);y=np.array([r['k_dynamic_real'] for r in sub]);kst=np.array([r['k_static'] for r in sub])
            line,=ax.plot(x,y,marker=mark,ms=4.2,lw=1.35,label=rf'$M\omega={om:.3f}$');lines.append((line,x,y,kst))
        ax.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$k_{20}^{\rm polar}(\omega)$',xlim=(0,1));ax.grid(alpha=.25);ax.legend(frameon=False)
        ins=inset_axes(ax,width='48%',height='43%',loc='center left',bbox_to_anchor=(.1,-.02,1,1),bbox_transform=ax.transAxes,borderpad=.8)
        for line,x,y,kst in lines: ins.plot(x,1e4*(y-kst),marker=line.get_marker(),ms=3,lw=1.05,color=line.get_color())
        ins.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$10^4[k(\omega)-k(0)]$',xlim=(0,1));ins.tick_params(labelsize=7);ins.grid(alpha=.22)
        fig.tight_layout();base=figs/f'{model}_polar_metric_dynamic_tln_vs_ell_ratio';fig.savefig(base.with_suffix('.png'),dpi=320);fig.savefig(base.with_suffix('.pdf'));plt.close(fig)
        
        fig,ax=plt.subplots(figsize=(7.2,5.0));flines=[]
        for ratio,mark in zip(FREQ_RATIOS,['o','s','^']):
            sub=sorted([r for r in frequency_rows if r['model']==model and abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda r:r['omega_M'])
            om=np.array([r['omega_M'] for r in sub]);y=np.array([r['k_dynamic_real'] for r in sub]);ks=np.array([r['k_static'] for r in sub]);line,=ax.plot(om,y,marker=mark,ms=3.8,lw=1.3,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$');flines.append((line,om,y,ks))
        ax.set(xlabel=r'$M\omega$',ylabel=r'$\mathrm{Re}\,k_{20}^{\rm polar}(\omega)$');ax.grid(alpha=.25);ax.legend(frameon=False,loc='center right')
        fins=inset_axes(ax,width='48%',height='43%',loc='lower left',bbox_to_anchor=(.08,.08,1,1),bbox_transform=ax.transAxes,borderpad=.8)
        for line,om,y,ks in flines:fins.plot(om,1e4*(y-ks),marker=line.get_marker(),ms=2.8,lw=1.0,color=line.get_color())
        fins.set(xlabel=r'$M\omega$',ylabel=r'$10^4[k(\omega)-k(0)]$');fins.tick_params(labelsize=7);fins.grid(alpha=.22)
        fig.tight_layout();base=figs/f'{model}_polar_metric_dynamic_tln_vs_frequency';fig.savefig(base.with_suffix('.png'),dpi=320);fig.savefig(base.with_suffix('.pdf'));plt.close(fig)
    summary={'models':MODELS,'charge_scan_frequencies':CHARGE_FREQS,'frequency_scan_ratios':FREQ_RATIOS,'frequency_range':[min(FREQS),max(FREQS)],'definition':'Direct reconstructed metric near-zone response with unit gravitational source and zero independent electromagnetic source. The independently recovered static response fixes the zero-frequency constant; no Schwarzschild subtraction is applied.','polar_love_normalization':'k20_polar=-(C_response/C_source)/M^5 in the metric convention used here','controlled_range':'Momega <= 0.006; 0.006 is the edge diagnostic and carries the largest matching-window uncertainty.'}
    (ROOT/'results'/'metric_scan_summary.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':
    c,f=run_all();plot(c,f)
