#!/usr/bin/env python3
"""Convergence table for the direct metric-level dynamical Love-number correction.
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATIO = 0.9
FREQUENCIES = (0.002, 0.004, 0.006)


def read_csv(path: Path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def emit(output, model, parity, omega, k_static, delta, residual):
    output.append({
        "model": model,
        "parity": parity,
        "ell_over_ell_ext": RATIO,
        "omega_M": omega,
        "k_dynamic_real": k_static + delta,
        "delta_k_real": delta,
        "window_max_residual": residual,
        "fractional_window_residual": residual / abs(delta),
    })


def main():
    output=[]
    # Bardeen polar: use the direct raw q=0.9 scan.
    bp=read_csv(ROOT/"metric_level"/"bardeen_polar_charge"/"results"/"raw_scans_recomputed"/"ellratio_0p90"/"results"/"bardeen_polar_dynamic_tln.csv")
    for r in bp:
        om=float(r["omega_M"])
        if om in FREQUENCIES:
            emit(output,"Bardeen","polar",om,float(r["k_static_reference"]),float(r["delta_k_real"]),float(r["window_max_residual"]))

    # Hayward/Fan-Wang polar: direct metric-ratio finite-frequency change on the same regular-black-hole background.
    hp=read_csv(ROOT/"metric_level"/"hayward_fanwang_polar"/"results"/"hayward_fanwang_metric_charge_scan.csv")
    for key,label in (("hayward","Hayward"),("fan_wang","Fan--Wang")):
        for r in hp:
            om=float(r["omega_M"])
            if r["model"]==key and abs(float(r["ell_over_ell_ext"])-RATIO)<1e-12 and om in FREQUENCIES:
                delta=-(float(r["metric_ratio_real"])-float(r["metric_ratio_static_real"]))
                emit(output,label,"polar",om,float(r["k_static"]),delta,float(r["window_max_residual"]))

    # Axial: solver stores raw_delta_real and k_unsubtracted_real directly.
    axial_sources=(
        ("Bardeen", ROOT/"metric_level"/"bardeen_axial"/"scan"/"q090"/"response.csv"),
        ("Hayward", ROOT/"metric_level"/"hayward_fanwang_axial"/"scan"/"hayward_q090"/"response.csv"),
        ("Fan--Wang", ROOT/"metric_level"/"hayward_fanwang_axial"/"scan"/"fan_wang_q090"/"response.csv"),
    )
    for label,path in axial_sources:
        for r in read_csv(path):
            om=float(r["omega_M"])
            if om in FREQUENCIES:
                emit(output,label,"axial",om,float(r["k_static"]),float(r["raw_delta_real"]),float(r["window_max_residual"]))

    model_order={"Bardeen":0,"Hayward":1,"Fan--Wang":2}
    parity_order={"polar":0,"axial":1}
    output.sort(key=lambda r:(parity_order[r["parity"]],model_order[r["model"]],r["omega_M"]))
    outdir=ROOT/"results"; outdir.mkdir(exist_ok=True)
    with (outdir/"dynamic_tln_convergence.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0].keys()))
        writer.writeheader(); writer.writerows(output)

if __name__=="__main__":
    main()
