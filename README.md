# Mass-conserving aerosols

Quick-and-dirty experiments to check if we can use a soft-max approach to emulate aerosol models.

## Original paper

Harder, Paula, et al. "Physics-informed learning of aerosol microphysics." Environmental Data Science 1 (2022): e20.   
[Link to pdf on arXiv](https://arxiv.org/pdf/2207.11786)  
[Link to paper on EDS](https://www.cambridge.org/core/journals/environmental-data-science/article/physicsinformed-learning-of-aerosol-microphysics/C468660D2AEE8E25DC3BF507517FF91A)
[Link to official code repository](https://github.com/paulaharder/aerosol-microphysics-emulation)

### Notes

- There are **4 aerosol species $s \in S$** (species is a domain term in this context), including SO4, BC, primary organic carbon (OC), and dust (DU).
- For are **7 modes for each species $m \in M$**, including N(-S), A-I, A-S, a-I, a-S, C-I, C-S. The first part of the mode stands for the size mode including N for nucleation, A for Aitken, a for accumulation, and C for coarse. While N practically only appear in the Solvable form, for A, a, and C, Insolvable and Solvable variants exist.
- 4 * 7 = 28 is the output dimensionality

Model:
- grid-cell-wise model (similar to PRESTO - comp. efficient)
- grid contains 31 vertical levels, 96 latitudes, and 192 longitudes. (~ 500k, i.e. 571392 grid cells for each t)
- x are the states with $x \in \mathbb{R}^{32}$ (28 + 4)
- y are the tendencies with $y \in \mathbb{R}^{28}$ (domain term for deltas i.e. $y_t = x_{t+1} - x_{t}$)


# Data

[Link to zenodo](https://zenodo.org/records/6583397)

We download it to the server with 
wget --content-disposition -P /path/to/dir "https://zenodo.org/records/6583397/files/aerosol_emulation_data.zip?download=1"

In my case /path/to/dir is /home/kim/data/aerosols. The data set is 4.3 GB so not small and took me ~10 minutes to download.

It contains the following .npy files, where validation and test are the same size: 
- X_test.npy
- X_train.npy
- X_val.npy
- y_test.npy
- y_train.npy
- y_val.npy

# Method

Softmax

## Ablations
- relative (proportional) inputs (mass per mode)
- absolute and relative inputs
    - potentially keep encoder seperate until a certain depth of the network (arcitecture ablation)
- check if geosppatial embedding like MOSAIK or SatCLIP help the prediction. While this is the atmosphere, satellite data may still contain useful information that are important for this.
- separate models for each s or shall we make it a meta-learning task.