# Dynamic TLN codes for Bardeen regular black holes

This repository contains two Python scripts accompanying the paper
**“Dynamical tidal response of regular black holes: Perturbative analysis and shell EFT interpretation.”**

The repository is intentionally focused on the Bardeen-sector numerical calculations:

1. `bardeen_dynamic_tln.py`  
   Computes frequency-dependent quadrupolar tidal Love-number scans for the Bardeen regular black hole using the coupled boundary-value problem, ingoing-wave boundary conditions, and hybrid far-field extraction.

2. `bardeen_shell_eft_comparison.py`  
   Compares the direct test-tensor response with the renormalized shell-EFT response for the Bardeen geometry in the low-frequency regime.

The paper also discusses Hayward and Fan--Wang geometries and additional coupled-response sectors. Those computations can be implemented analogously by replacing the Bardeen metric functions and matter-sector coefficients with the corresponding background quantities.

## Requirements

The scripts require Python 3.10 or later and the packages listed in `requirements.txt`.

Install dependencies with

```bash
python -m pip install -r requirements.txt
```

## Usage

### 1. Bardeen dynamical TLN scan

Quick run with small grids:

```bash
python bardeen_dynamic_tln.py --mode length --ell-ratios linspace:0.01,0.30,3 --omegas 0.0001 --output-dir results --figure-dir figures
```

Default full run:

```bash
python bardeen_dynamic_tln.py
```

The script writes CSV outputs to `results/` and figures to `figures/`.

### 2. Direct test-tensor versus shell-EFT comparison

Quick non-plot test:

```bash
python bardeen_shell_eft_comparison.py --n-chi 3 --omega 1e-4 --no-plot
```

Default comparison run:

```bash
python bardeen_shell_eft_comparison.py
```

## Numerical notes

The calculations involve boundary-value and high-accuracy ODE solves. Full production scans can be computationally heavy. For checking installation and reproducibility, start with the quick commands above.

The response normalizations follow the conventions used in the original research scripts. In particular, the direct Bardeen dynamical TLN script reports the normalized response used for plotting in the manuscript.

## Citation

If you use this repository, please cite the accompanying paper and this GitHub repository.

## License

This code is released under the MIT License. See `LICENSE`.
