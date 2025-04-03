# Mass-conserving aerosol microphysics

Quick-and-dirty one-day experiments to check if we can leverage a reformulation of the grid-cell-wise mass conservation problem that let's us apply a NN with soft-max output layer as the aerosol emulation model, while **guaranteeing non-negativity as well as mass conservation**.

The tendency $R^2$ score reported in [the paper](https://arxiv.org/pdf/2207.11786) is 77.1% (0.77; average over all 4 aerosol species). 

With very limited training and no conditioning on location or time we already a mean $R^2$ of 0.945 and the following $R^2$ scores per aerosol species on the provided test data sets:
- SO4: 0.849
- BC: 0.973
- OC (not California...): 0.963
- DU: 0.997

Each epoch trains over all 5 M training points.

However, the tendency $R^2$ is still high as tendencies are small.

We now rather parameterise the transition matrix! The rows of the transition matrix are here parameterised with softmax outputs!

These results can be reporduced by running *main_notebook.ipynb*

## Original paper

Harder, Paula, et al. "Physics-informed learning of aerosol microphysics." Environmental Data Science 1 (2022): e20.   
[Link to pdf on arXiv](https://arxiv.org/pdf/2207.11786)  
[Link to paper on EDS](https://www.cambridge.org/core/journals/environmental-data-science/article/physicsinformed-learning-of-aerosol-microphysics/C468660D2AEE8E25DC3BF507517FF91A)  
[Link to official code repository](https://github.com/paulaharder/aerosol-microphysics-emulation)

### Notes

- There are **4 aerosol species $s \in S$** (species is a domain term in this context), including SO4, BC, primary organic carbon (OC), and dust (DU).
- For are **7 modes for each species $m \in M$**, including N(-S), A-I, A-S, a-I, a-S, C-I, C-S. The first part of the mode stands for the size mode including N for nucleation, A for Aitken, a for accumulation, and C for coarse. While N practically only appear in the Solvable form, for A, a, and C, Insolvable and Solvable variants exist.
- 4 * 7 = 28 is the output dimensionality (Question: Only 24 are included and 17 are used. Why?)

Model:
- grid-cell-wise model (similar to PRESTO - comp. efficient)
- grid contains 31 vertical levels, 96 latitudes, and 192 longitudes. (~ 500k, i.e. 571392 grid cells for each t)
- x are the states with $x \in \mathbb{R}^{32}$ (28 + 4)
- y are the tendencies with $y \in \mathbb{R}^{28}$ (domain term for deltas i.e. $y_t = x_{t+1} - x_{t}$)

Code:
- The experiments use only 24 columns, not 28.
- For X which has 32 dims we use the last 24 columns
- For y which has 28 dims we use the first 24 columns only
- Out of these 24 columns, only 17 are used. 5 for SO4, and 4 for BC, OC, and DU

# Data

[Link to zenodo](https://zenodo.org/records/6583397)

We download it to the server with 
wget --content-disposition -P /path/to/dir "https://zenodo.org/records/6583397/files/aerosol_emulation_data.zip?download=1"

In my case /path/to/dir is /home/kim/data/aerosols. The data set is 4.3 GB so not small and took me ~10 minutes to download.

It contains the following .npy files, where validation and test are the same size: 
- X_test.npy shape (2856955, 32) 2.8M cells
- X_train.npy
- X_val.npy
- y_test.npy shape (2856955, 28)
- y_train.npy
- y_val.npy

# Proposed Method: Relative mass distribution over modes using soft-max output and species-specific models

As the total mass per grid cells stays the same, we use the relative (discrete) mass distribution over the 4 (or 5) modes at t = 0 as input (x), and predict the new mass distribution at t = 1 (y). Using the total input mass (which should be the same as the total output mass under mass conservation) we project back from relative to absolute mass values. The softmax ensures that masses are non-negative and that the relative proportions add to one. We build one model per aerosol species to enable this. 

## Ablations
- relative (proportional) inputs (mass per mode)
- absolute and relative inputs
    - potentially keep encoder seperate until a certain depth of the network (arcitecture ablation)
- check if geosppatial embedding like MOSAIK or SatCLIP help the prediction. While this is the atmosphere, satellite data may still contain useful information that are important for this.
- separate models for each s or shall we make it a meta-learning task.

## Notes to self
- Is a transition framework more stable as an approach if we do **multi-task learning**?
    - Can a Neural Process understand which subset we are in?
- Learnings:
    - predict y_delta not y
- LogSoftmax_model
    - log transform input x
    - predict tendencies directly
    - avoid negativity through scaling
    - scaled zero-mean softmax is used
