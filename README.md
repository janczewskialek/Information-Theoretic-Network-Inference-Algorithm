# Information-Theoretic Network Inference Algorithm

**Author:** Aleksander Janczewski  
**Date:** 6th August 2023  
**Version:** 2.0

## Overview
This project offers two primary functionalities: Information-Theoretic Network Inference and Information Percolation for VECM models, each with its distinct set of features.

### /TransferEntropy

The C++ source code for an information-theoretic network inference algorithm. This algorithm leverages the KSG algorithm I for estimating continuous apparent and conditional transfer entropies based on the methodologies of Kraskov et al. [2004], Frenzel and Pompe [2007], Ragwitz and Kantz [2002] and Wibral et al. [2013].

### /InformationPercolation

Provides C++ source code for a VECM model that computes information share as introduced by Hasbrouck [1995] and price inefficiency metrics from Hagstromer and Menkveld [2019]. The model incorporates the BIC lag selection and employs the least squares method for fitting VECM to time series data.

## Features
- **KSG Algorithm I**: Continuous and conditional transfer entropy estimations.
- **Embedding and Delays**: Derived using the Ragwitz criterion (Ragwitz and Kantz [2002]).
- **True Delay Detection**: Based on the methodology of Wibral et al. [2013].
- **Permutation Testing**: For statistical significance assessment.
- **Information Share**: Calculates dealer-wise (market) information share.
- **Price Inefficiency**: Dealer-wise (market) price inefficiency computation.

## Requirements

### C++ Dependencies:
- C++17
- Clang 13.0.0 / GCC 11.2.0
- Eigen 3.4.0 (with unsupported Matrix functions module)
- CMake 3.2.0

### Python Dependencies:
Refer to the `requirements.txt` file for Python package prerequisites.

## Directory Structure

| Directory | Description |
| --- | --- |
| /TransferEntropy | Main source code for information-theoretic network inference |
| /TransferEntropy/INA.cpp | Main implementation of the network inference algorithm |
| /TransferEntropy/ckdtree | Modified [Scipy's ckdtree](https://github.com/scipy/scipy/tree/main/scipy/spatial/ckdtree) |
| /InformationPercolation | Main source code for the econometric model and Python/Cython wrapper |
| /InformationPercolation/code/FX_infoperc | C++ (`/code_cpp`) and Cython (`/code_cython`) sources for the model and wrapper |
| /circ_shift.h | [External function](https://stackoverflow.com/questions/46077242/eigen-modifyable-custom-expression/46301503#46301503) for array rolling |

## Building and Using the Python Library

1. Navigate to `InformationPercolation/code/FX_infoperc`.
2. Execute: `python setup.py build_ext --inplace clean --all`
   
This will compile the C++ code and produce a Python library wrapper. Once generated, this can be imported into Jupyter notebooks or any Python module. For a practical demonstration, see the `Example/Example1.ipynb` Jupyter notebook.

---
