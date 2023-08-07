"""
This module contains a container that can store results of the analysis.
"""

import numpy as np
import numpy
import subprocess
import os
import seaborn as sns


class VECM_result(object):
    """Container to store results of the analysis. It can draw network maps as well.
    """
    def __init__(self,
                 S_lower=None,
                 S_upper=None,
                 long_run=None,
                 component_share=None,
                 edges=None,
                 edges_corr=None,
                 residuals=None,
                 alpha_coeff=None,
                 pi=None,
                 gamma_coeff=None,
                 theta_coeff=None,
                 covariance=None,
                 y_ast=None,
                 z_est=None,
                 included=None,
                 dY=None,
                 dX=None,
                 Y_1=None,
                 beta_inefficiency=None,
                 TE = None,
                 TE_pval = None,
                 CTE = None,
                 CTE_pval = None):

        self.S_lower = S_lower.flatten()
        self.S_upper = S_upper.flatten()
        self.long_run = long_run.flatten()
        self.component_share = component_share.flatten()
        self.edges = edges
        self.edges_corr = edges_corr
        self.residuals = residuals
        self.alpha_coeff = alpha_coeff
        self.pi = pi
        self.gamma_coeff = gamma_coeff
        self.theta_coeff = theta_coeff
        self.covariance = covariance
        self.y_ast = y_ast
        self.z_est = z_est
        self.included = included.flatten()
        self.dY = dY
        self.dX = dX
        self.Y_1 = Y_1
        self.beta_inefficiency = beta_inefficiency
        self.TE = TE
        self.TE_pval = TE_pval
        self.CTE = CTE
        self.CTE_pval = CTE_pval

        self.n = len(self.included)
