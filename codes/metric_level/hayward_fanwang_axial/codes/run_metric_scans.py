#!/usr/bin/env python3
from __future__ import annotations
import csv,time
from pathlib import Path
from rbh_axial_metric_solver import solve
ROOT=Path(__file__).resolve().parents[1];SCAN=ROOT/'scan';SCAN.mkdir(exist_ok=True)
RATIOS=[0.,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
CHARGE_FREQS=[.002,.004,.006];FREQ_RATIOS={.5,.8,.95};FREQ_GRID=[.001,.002,.003,.004,.005,.006]

def main():
    for model in ('hayward','fan_wang'):
        for ratio in RATIOS:
            freqs=FREQ_GRID if ratio in FREQ_RATIOS or ratio==0. else CHARGE_FREQS
            out=SCAN/f'{model}_q{int(round(ratio*100)):03d}';out.mkdir(parents=True,exist_ok=True);t=time.time()
            rows,wins=solve(model,ratio,freqs)
            for name,data in [('response.csv',rows),('windows.csv',wins)]:
                with (out/name).open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            print('done',model,ratio,'seconds',round(time.time()-t,2),flush=True)
if __name__=='__main__':main()
