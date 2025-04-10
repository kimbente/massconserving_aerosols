# Mass-conserving aerosol microphysics

Quick-and-dirty three-day experiments to check if we can leverage a reformulation of the grid-cell-wise mass conservation problem that let's us apply a NN with soft-max output layer (i.e. a transition matrix) as the aerosol emulation model, while **guaranteeing non-negativity as well as mass conservation**.

The tendency $R^2$ score reported in [the paper](https://arxiv.org/pdf/2207.11786) is 77.1% (0.77; average over all 4 aerosol species). 

With very limited training and no conditioning on location or time we achive a mean $R^2$ of 0.945 and the following $R^2$ scores per aerosol species on the provided test data sets, however this was not using tendencies but x (at t = 0) and y (at t = 1).
- SO4: 0.849
- BC: 0.973
- OC (not California...): 0.963
- DU: 0.997

However, the tendency $R^2$ is still high as tendencies are small. These results can be reporduced by running *main_notebook.ipynb*

## These were my ideas:

All ideas rely on **hard-constraining** the problem directly, which requires building a model for ever species.
- **Transition matrix**
    - Estimating an n_classes x n_classes transition matrix where we can use row-wise softmax outputs to ensure that all mass is conserved. 
    - Then the transition proprotions (not quite probabilities here) are applied to the absolute masses at t = 0 via multiplication, to get the transitioning masses (tendendies) while ensuring non-negativity. (If a "mode" is mass 0, multiplying anything by it will still yield zero.)
    - The tendencies are very small compared to the total mass, particularly for SO4, which is why - even with stabilising the transition matrix parameterision which a strong diagonal self-transition bias, it might have issues in this case. Let's run some experiments.
    - This requires a zero preserving transformation of the input, so that non-negativity can be guaranteed. Arcsin might be better than log (while clamping negative values as they technically are not allowed.)
    - Technically this is the more elegant method as both constraints are met by deisgn. However we need to make it work with the correct transformations. Parsimony is key.
    - We take the total and relative input explicitly so the network can use both information as input.
    - One output of the network is used to scale the ouput tendencies properly.
    - We leverage additional variables to get better predictions. 

- **Predicting mass conserving tendencies** (old approach)
    - By predicting a tendency vector that adds to zero we can preserve mass. 
    - This can be achieved my using any output (or a softmax output because it has a known mean (1/length)), subtracting the (known) mean so we get a zero-mean vector which is **mass conserving**. However with non-linear transformations applied to the output, like arcsin(), this is not straightforward.
    - To achieve **non-negativity**, we can apply a linear scaling to the tendency vector in the last step. Using the OG input scale, we can determine what scaling factor is needed to satisfy the constraint. In the worst case the vector is multiplied by 0. Linear scaling does not distort the property. 

Design decisions:
- Transformations (really important here!)
    - log - does not preserve zero, and is not even defined for zero.
    - arcsinh - preserve zero and is thus preferrable
- target
    - tendencies (i.e. deltas) or totals
    - deltas are very small in relation to totals, so to get any learning signal we should rather estimate the tendencies (and to compare to the work we are building on.)
    - scales

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
- Model per species as data patterns are different.
- Does the model in the original paper benefit from the **additional inputs**? If we use all 24 inputs to estimate one species (e.g. 5 outputs), is that better?
- How big is the difference between 
- Each epoch trains over all 5 M training points.

## ToDos
- add relative signal (test of all)
    - remove rows with issue
- experiment with bias term
- use R^2 as the loss

