# Information-Theoretic Network Inference Algorithm

Author: Aleksander Janczewski

Date: 24th July 2022

Version: 1.0

# Overview
The project contains the C++ source code for an information-theoretic network inference algoritm. The algorithm utilizes the KSG algorithm I to estimate continuous transfer and conditional transfer entropies following the methodology of (Kraskov et al. [2004]) and (Frenzel and Pompe [2007]), respectively. The embedding history and delays for the KSG algorithm are determined using the Ragwitz criterion (Ragwitz and Kantz, [2002]). Furthermore, the true delay between source and target processes are determined following the method proposed by (Wibral et al. [2013]). Finally, permutation testing is performed to determine the statistical significance of the estimates.


**Functions available:**
- KSG algorithm I
- Automatic and constant delay and history embedding for all processes based on Ragwitz criterion.
- Automatic detection and constant source-target delay for all processes based on entropy maximization condition.
- Number of permutations in permutation testing
- Minimum statistical significance 


# Requirements
- C++17
- Clang 13.0.0 
- Eigen 3.4.0 with unsupported Matrix functions module
- CMake 3.2.0


| Directory          | Description                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------ |
| /INA.cpp           | The main source code with information-theoretic network inference algorithm                      |
| /ckdtree           | Directory contains the source code for [Scipy's ckdtree](https://github.com/scipy/scipy/tree/main/scipy/spatial/ckdtree) modified for the purposes of this project.|
| /circ_shift.h      | [External function](https://stackoverflow.com/questions/46077242/eigen-modifyable-custom-expression/46301503#46301503) used for rolling arrays                                                  |
