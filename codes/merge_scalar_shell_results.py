#!/usr/bin/env python3
from __future__ import annotations
import csv, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
import os
PART=Path(os.environ.get('SCALAR_PARTIAL_DIR', ROOT/'partial_results'))
MODELS=('bardeen','hayward','fan_wang')
OMEGAS=(1e-4,2e-4,5e-4)
rows=[]; summary={}
schema=None
for model in MODELS:
    with (PART/f'{model}.csv').open() as f:
        reader=csv.DictReader(f)
        if schema is None:
            schema=reader.fieldnames
        elif reader.fieldnames != schema:
            raise RuntimeError(f'Inconsistent scalar CSV schema for {model}')
        model_rows=list(reader)
        if len(model_rows) != 27:
            raise RuntimeError(f'Expected 27 scalar rows for {model}, found {len(model_rows)}')
        if {row['model'] for row in model_rows} != {model}:
            raise RuntimeError(f'Incorrect model label in the {model} scalar partial')
        for row in model_rows:
            converted={}
            for k,v in row.items():
                converted[k]=v if k=='model' else float(v)
            rows.append(converted)
    summary.update(json.load(open(PART/f'{model}.json')))
if len(rows) != 81:
    raise RuntimeError(f'Expected 81 merged scalar rows, found {len(rows)}')
if {row['model'] for row in rows} != set(MODELS):
    raise RuntimeError('Merged scalar output does not contain all three models')
if set(summary) != set(MODELS):
    raise RuntimeError('Merged scalar JSON does not contain all three models')
results=ROOT/'results'; results.mkdir(exist_ok=True)
for name in ('scalar_shell_eft_all_models.csv', 'scalar_shell_eft_all_models_direct.csv'):
    with (results/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
encoded=json.dumps(summary,indent=2)+'\n'
for name in ('scalar_shell_eft_all_models.json', 'scalar_shell_eft_all_models_direct.json'):
    (results/name).write_text(encoded)

fig,axes=plt.subplots(3,2,figsize=(10.8,11.5))
labels={'bardeen':'Bardeen','hayward':'Hayward','fan_wang':'Fan--Wang'}
for i,model in enumerate(MODELS):
    for om in OMEGAS:
        sel=sorted([r for r in rows if r['model']==model and np.isclose(r['omega_M'],om)],key=lambda r:r['ell_over_ell_ext'])
        x=np.array([r['ell_over_ell_ext'] for r in sel]);direct=np.array([r['direct_real'] for r in sel]);shell=np.array([r['shell_real'] for r in sel])
        line=axes[i,0].plot(x,direct,'o-',ms=3.3,lw=1.25,label=rf'direct, $M\omega={om:g}$')[0]
        axes[i,0].plot(x,shell,'--',lw=1.35,color=line.get_color(),label=rf'shell, $M\omega={om:g}$')
        axes[i,1].plot(x,shell-direct,'o-',ms=3.3,lw=1.25,color=line.get_color(),label=rf'$M\omega={om:g}$')
    axes[i,0].set_title(labels[model])
    axes[i,0].set_ylabel(r'renormalized scalar response $\Delta\Lambda_0$')
    axes[i,1].set_ylabel('shell minus direct')
    for ax in axes[i]:
        ax.axhline(0,lw=.7,color='0.5');ax.set_xlabel(r'$\ell/\ell_{\rm ext}$');ax.grid(alpha=.22);ax.legend(frameon=False,fontsize=7.2)
fig.suptitle('Scalar shell EFT with same-background source subtraction',y=.995)
fig.tight_layout()
figdir=ROOT.parent/'figures';figdir.mkdir(exist_ok=True)
fig.savefig(figdir/'scalar_shell_eft_all_models.png',dpi=240)
fig.savefig(figdir/'scalar_shell_eft_all_models.pdf')
plt.close(fig)
