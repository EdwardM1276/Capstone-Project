# Data

Place the original PaySim CSV at `data/raw/paysim.csv`. The dataset is excluded
from Git because the supplied file is approximately 493 MB and should not be
redistributed with this repository. The application uses the PaySim schema:

`step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`

The preprocessing and training commands validate the complete source file.
Training identifiers are not loaded as predictive features.

## Review sample datasets

The twelve CSVs in `sample_datasets/` each contain 100,000 transactions from
the original PaySim dataset. They are separate, non-overlapping samples and
are ordered by `step`. The generator reproduces the project's seeded,
stratified 200,000-row training sample and excludes those exact source rows
before selecting any review transactions. Each generated file is about 7.8 MB,
below the app's 200 MB upload limit. The original source file is not included
in Git. Download a CSV to your computer from this folder in the GitHub
repository, then select that local file with the app's CSV upload button; the
app does not preload or offer these samples in a dropdown.

Each sample retains the PaySim `isFraud` label for aggregate benchmark
estimates in the cost simulator. The label is not a model input and is hidden
from the analyst queue and per-alert details.

To regenerate the files from the original CSV in PowerShell, run this from
the repository root:

```powershell
python .\scripts\create_review_samples.py --source "C:\path\to\paysim.csv"
```

The default output is `data/sample_datasets/`. The command refuses to overwrite
an output directory that already contains files. `dataset_manifest.json`
records the source row count, seeds, file sizes, fraud counts, and step ranges.
