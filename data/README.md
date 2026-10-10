# Data

Download `PS_20174392719_1491204439457_log.csv` from
[Kaggle's Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
and place it at `data/raw/paysim.csv`. The dataset is excluded from Git because
it is approximately 493 MB. It is synthetic and should be described as a
simulator dataset, not as a sample of real customer transactions. The
application uses the PaySim schema:

`step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`

The preprocessing and training commands validate the complete source file.
Training identifiers are excluded from predictive features, and quality
reports record the source filename rather than a machine-specific absolute
path.
