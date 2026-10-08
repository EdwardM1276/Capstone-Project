# Data

Place the original PaySim CSV at `data/raw/paysim.csv`. The dataset is excluded
from Git because the supplied file is approximately 493 MB and should not be
redistributed with this repository. The application uses the PaySim schema:

`step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud`

The preprocessing and training commands validate the complete source file.
Training identifiers are not loaded as predictive features.
