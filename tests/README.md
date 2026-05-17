# QMaxent test suite

The test suite is split into two tiers:

```
tests/
├── unit/         # Pure-Python logic that does not touch QGIS / Qt.
│                 # Always runnable from a plain Python install with
│                 # numpy + rasterio + scikit-learn — no QGIS required.
├── regression/   # End-to-end claims made in the SoftwareX manuscript
│                 # (§ 3.3 maxent.jar ↔ QMaxent numerical compatibility).
│                 # Compared against `tests/fixtures/*.json` baselines
│                 # with documented tolerances.
└── fixtures/     # Small synthetic / sub-sampled inputs + JSON-encoded
                  # "golden" outputs used by the regression tier.
```

## Running the tests

From the repository root:

```bash
# 1. Install runtime + test dependencies into a venv.
python -m venv .venv
.venv/bin/activate          # or  .venv\Scripts\activate on Windows
pip install -r requirements-test.txt

# 2. Run only the unit tier (fast, QGIS-free, runs on every push).
pytest tests/unit -q

# 3. Run the regression tier (slower, requires elapid + rasterio).
pytest tests/regression -q

# 4. Or run everything with coverage.
pytest --cov=. --cov-report=term-missing
```

## What is intentionally NOT tested here

- **QGIS-internal dialogs and widgets**. The UI layer (`dialogs/`,
  `qmaxent_plugin.py`, the QThread `workers/`) drives QGIS / Qt
  objects that cannot be instantiated outside a running QGIS
  process. These layers are validated manually (the manuscript's
  Fig. 2 panels (a)–(l) are the visual contract) and via the
  end-to-end Bradypus and Pitta nympha examples shipped with the
  plugin.
- **maxent.jar itself**. The maxent.jar comparison numbers in
  `tests/fixtures/pitta_golden_values.json` were produced offline
  and are checked against; we do not invoke maxent.jar from CI
  (no Java setup, no licensing complications).

## Adding new tests

Most new tests belong in `tests/unit/`. A test belongs in
`tests/regression/` if and only if it directly defends a quantitative
claim that the manuscript makes about output values.

Each regression test must:

1. Document the manuscript claim it defends (section, table, figure).
2. Pin a small fixture under `tests/fixtures/` (no large data;
   downsample if needed).
3. Specify the tolerance and explain why (e.g., "|Δ| < 0.005 for
   training AUC, matching the IWLR ↔ coordinate-descent micro-
   convergence band reported in § 2.3").
