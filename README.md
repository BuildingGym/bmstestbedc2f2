# bmstestbedc2f2

## Usage
- `packages/bmstestbedc2f2_notebooks/run_eplus.ipynb`: train the model using the EnergyPlus simulation and save the model to disk; this overrides previously saved model.
- `packages/bmstestbedc2f2_notebooks/run_manual.ipynb`: load the model from disk and train the trained EnergyPlus model through real-world interactions; this updates the saved model.

## Setup
```sh
python3 -m pip install -e .
```

```sh
# TODO
# python3 -m pip install controllables-core
# --extra-index-url https://test.pypi.org/simple 
# git+https://github.com/NTU-CCA-HVAC-OPTIM-a842a748/EnergyPlus-OOEP
```