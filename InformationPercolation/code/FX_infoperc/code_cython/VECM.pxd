"""
This code specifies available functions and variables in the
VECM.h and the VECM.cpp files.
"""

# We must include this file
cdef extern from "VECM.cpp":
    pass

# List all functions to make them available for Python
cdef extern from "VECM.h":
    cdef cppclass VECM:
        # Constructors
        VECM() except +
        VECM(double*, int, int, int, int, int, int, int, int, int) except +

        # Fit
        void run_models(int)

        # Output
        int get_num_included()
        int get_dZ_num()

        double* get_S_lower_ptr()
        double* get_S_upper_ptr()
        double* get_long_run_ptr()
        double* get_component_share_ptr()
        double* get_edges_ptr()
        double* get_edges_corr_ptr()
        double* get_covariance_ptr()
        double* get_data_ptr()
        int* get_included_ptr()
        double* get_dZ_ptr()
        double* get_residuals_ptr()

        double* get_alpha_ptr()
        double* get_pi_ptr()
        double* get_beta_ptr()
        double* get_gamma_ptr()
        double* get_theta_ptr()
        double* get_beta_ineff_ptr()

        double* get_TE_ptr()
        double* get_TE_pval_ptr()
        double* get_CTE_ptr()
        double* get_CTE_pval_ptr()

        void print_all()









