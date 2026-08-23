# Reproducibility checks

Run the complete calculation with

```bash
python3 run_all.py
```

The release uses `codes/einstein_nled_master.py` as the only canonical
Einstein--NLED master solver.  The radial integration cap is an explicit
`SolverControls.max_step_radius` parameter, so workflows that previously used
different vendored copies retain their original integration settings without
path-dependent imports.

The canonical direct scalar outputs are
`codes/results/scalar_shell_eft_all_models_direct.csv` and
`codes/results/scalar_shell_eft_all_models_direct.json`.  They contain all
three models.  The CSV has 81 rows, with 27 rows for each model.

`release_manifest.json` contains the exact machine-readable cells for every
manuscript table input and, for every manuscript figure, the command, producing
script, archived parameter grids, response convention, and code, input, and
output SHA-256 hashes.  Verify the frozen release values with

```bash
python3 codes/release_manifest.py --verify
```

The relative peak-response change is
[
rac{|mathcal R_{mathrm{strict}}-mathcal R_{mathrm{standard}}|}
     {|mathcal R_{mathrm{strict}}|},
]
evaluated at the frequency where the standard-control response magnitude
reaches its sampled peak.
