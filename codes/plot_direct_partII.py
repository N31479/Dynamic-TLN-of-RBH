#!/usr/bin/env python3
from __future__ import annotations
import csv, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

ROOT = Path(__file__).resolve().parent.parent
CODES = ROOT / 'codes'
FIG = ROOT / 'figures'
RES = CODES / 'results' / 'direct_metric'
FIG.mkdir(exist_ok=True)
RES.mkdir(parents=True, exist_ok=True)


def read(path):
    rows=[]
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            d={}
            for k,v in r.items():
                try:d[k]=float(v)
                except:d[k]=v
            rows.append(d)
    return rows


def save_rows(name, rows):
    if not rows:return
    with open(RES/name,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def plot_pair(model_label, parity, charge_rows, freq_rows, charge_freqs=(.002,.004,.006), freq_ratios=(.5,.9,.95)):
    # charge scan
    fig,ax=plt.subplots(figsize=(7.2,5.0))
    ratios=sorted(set(r['ell_over_ell_ext'] for r in charge_rows))
    stat=[]
    for x in ratios:
        rr=[r for r in charge_rows if abs(r['ell_over_ell_ext']-x)<1e-12]
        stat.append(rr[0]['k_static'])
    ax.plot(ratios,stat,'--',lw=1.8,label=r'Static ($M\omega=0$)')
    lines=[]
    for om,mark in zip(charge_freqs,['o','s','^']):
        rr=sorted([r for r in charge_rows if abs(r['omega_M']-om)<1e-12],key=lambda z:z['ell_over_ell_ext'])
        x=np.array([r['ell_over_ell_ext'] for r in rr]); y=np.array([r['k_direct_real'] for r in rr]); ks=np.array([r['k_static'] for r in rr])
        line,=ax.plot(x,y,marker=mark,ms=4.0,lw=1.3,label=rf'$M\omega={om:.3f}$');lines.append((line,x,y,ks))
    ax.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=rf'$k_{{20}}^{{\rm {parity}}}(\omega)$');ax.grid(alpha=.24);ax.legend(frameon=False)
    ins=inset_axes(ax,width='48%',height='42%',loc='center left',bbox_to_anchor=(.10,-.02,1,1),bbox_transform=ax.transAxes,borderpad=.8)
    for line,x,y,ks in lines:ins.plot(x,1e4*(y-ks),marker=line.get_marker(),ms=2.8,lw=1.0,color=line.get_color())
    ins.set(xlabel=r'$\ell/\ell_{\rm ext}$',ylabel=r'$10^4[k(\omega)-k(0)]$');ins.tick_params(labelsize=7);ins.grid(alpha=.2)
    fig.tight_layout(); stem=f"{model_label.lower().replace('–','_').replace('--','_').replace('-','_').replace(' ','_')}_{parity}_metric_dynamic_tln_vs_ell_ratio";fig.savefig(FIG/(stem+'.png'),dpi=300);fig.savefig(FIG/(stem+'.pdf'));plt.close(fig)
    # frequency scan
    fig,ax=plt.subplots(figsize=(7.2,5.0)); lines=[]
    available=sorted(set(r['ell_over_ell_ext'] for r in freq_rows))
    wanted=[min(available,key=lambda a:abs(a-q)) for q in freq_ratios]
    # unique preserving order
    seen=[]; wanted=[x for x in wanted if not (x in seen or seen.append(x))]
    for ratio,mark in zip(wanted,['o','s','^']):
        rr=sorted([r for r in freq_rows if abs(r['ell_over_ell_ext']-ratio)<1e-12],key=lambda z:z['omega_M'])
        if not rr:continue
        om=[0.]+[r['omega_M'] for r in rr]
        y=[rr[0]['k_static']]+[r['k_direct_real'] for r in rr]
        ks=[rr[0]['k_static']]*len(om)
        line,=ax.plot(om,y,marker=mark,ms=3.6,lw=1.25,label=rf'$\ell/\ell_{{\rm ext}}={ratio:.2f}$');lines.append((line,om,y,ks))
    ax.set(xlabel=r'$M\omega$',ylabel=rf'$\mathrm{{Re}}\,k_{{20}}^{{\rm {parity}}}(\omega)$');ax.grid(alpha=.24);ax.legend(frameon=False)
    ins=inset_axes(ax,width='48%',height='42%',loc='lower left',bbox_to_anchor=(.08,.08,1,1),bbox_transform=ax.transAxes,borderpad=.8)
    for line,om,y,ks in lines:ins.plot(om,1e4*(np.array(y)-np.array(ks)),marker=line.get_marker(),ms=2.6,lw=1.0,color=line.get_color())
    ins.set(xlabel=r'$M\omega$',ylabel=r'$10^4[k(\omega)-k(0)]$');ins.tick_params(labelsize=7);ins.grid(alpha=.2)
    fig.tight_layout();stem=f"{model_label.lower().replace('–','_').replace('--','_').replace('-','_').replace(' ','_')}_{parity}_metric_dynamic_tln_vs_frequency";fig.savefig(FIG/(stem+'.png'),dpi=300);fig.savefig(FIG/(stem+'.pdf'));plt.close(fig)

# Bardeen polar charge: absolute response is explicitly stored.
bp = read(CODES/'metric_level/bardeen_polar_charge/results/bardeen_polar_metric_dynamic_tln_charge_scan_long.csv')
bp_charge=[]
for r in bp:
    bp_charge.append({'model':'bardeen','ell_over_ell_ext':r['ell_over_ell_ext'],'omega_M':r['Momega'],'k_static':r['k_static'],'k_direct_real':r['k_dynamic_real'],'k_direct_imag':r['k_dynamic_imag'],'raw_delta_real':r['delta_k_real']})
# raw per-ratio files provide the direct frequency continuation
bp_freq=[]
base=CODES/'metric_level/bardeen_polar_charge/results/raw_scans_recomputed'
for ratio,tag in [(0.50,'ellratio_0p50'),(0.90,'ellratio_0p90'),(0.95,'ellratio_0p95')]:
    for r in read(base/tag/'results/bardeen_polar_dynamic_tln.csv'):
        bp_freq.append({'model':'bardeen','ell_over_ell_ext':ratio,'omega_M':r['omega_M'],'k_static':r['k_static_reference'],'k_direct_real':r['k_dynamic_real'],'k_direct_imag':r['k_dynamic_imag'],'raw_delta_real':r['delta_k_real']})
save_rows('bardeen_polar_unsubtracted.csv',bp_charge)
plot_pair('Bardeen','polar',bp_charge,bp_freq,freq_ratios=(.5,.9,.95))

# Hayward/Fan-Wang polar: use the direct metric response stored by the production scan.
hfp_c=read(CODES/'metric_level/hayward_fanwang_polar/results/hayward_fanwang_metric_charge_scan.csv')
hfp_f=read(CODES/'metric_level/hayward_fanwang_polar/results/hayward_fanwang_metric_frequency_scan.csv')
for model,label in [('hayward','Hayward'),('fan_wang','Fan_Wang')]:
    cr=[]
    for r in hfp_c:
        if r['model']!=model:continue
        raw=r['delta_real']
        cr.append({**r,'k_direct_real':r['k_dynamic_real'],'raw_delta_real':raw})
    fr=[]
    for r in hfp_f:
        if r['model']!=model or r['omega_M']==0:continue
        raw=r['delta_real']
        fr.append({**r,'k_direct_real':r['k_dynamic_real'],'raw_delta_real':raw})
    save_rows(f'{model}_polar_unsubtracted.csv',cr)
    plot_pair(label,'polar',cr,fr,freq_ratios=(.5,.9,.97))

# Axial: solver already stores k_unsubtracted_real.
ba=[]
for d in sorted((CODES/'metric_level/bardeen_axial/scan').glob('q*')):
    p=d/'response.csv'
    if p.exists():ba += read(p)
ba2=[{**r,'k_direct_real':r['k_unsubtracted_real']} for r in ba]
save_rows('bardeen_axial_unsubtracted.csv',ba2)
plot_pair('Bardeen','axial',ba2,ba2,freq_ratios=(.5,.8,.95))

hfa=[]
for d in sorted((CODES/'metric_level/hayward_fanwang_axial/scan').glob('*_q*')):
    p=d/'response.csv'
    if p.exists():hfa += read(p)
for model,label in [('hayward','Hayward'),('fan_wang','Fan_Wang')]:
    rr=[{**r,'k_direct_real':r['k_unsubtracted_real']} for r in hfa if r['model']==model]
    save_rows(f'{model}_axial_unsubtracted.csv',rr)
    plot_pair(label,'axial',rr,rr,freq_ratios=(.5,.8,.95))

print(f'Wrote direct Part II figures to {FIG}')
