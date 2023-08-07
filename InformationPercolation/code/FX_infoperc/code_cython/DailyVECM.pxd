"""
This code specifies available functions and variables in the
VECM.h and the VECM.cpp files.
"""

# We need to include this
cdef extern from "DailyVECM.cpp":
    pass

# The header file. The function listed here will be available in Python
cdef extern from "DailyVECM.h":
    cdef cppclass DailyVECM:
        # Constructors
        DailyVECM() except +
        DailyVECM(int*, int*, double*, double*, int) except +

        # Init and run
        void init(int, int, int, int, int)
        int run(int, int, int, int, int, int, int, int ,int )


        # Getters to get the pointers to the first element in the matrices
        double* get_time_ptr()
        int* get_included_ptr()
        double* get_spread_ptr()
        double* get_variance_ptr()
        double* get_S_lower_ptr()
        double* get_S_upper_ptr()
        double* get_long_run_ptr()
        double* get_component_share_ptr()
        double* get_edges_ptr()
        double* get_edges_corr_ptr()
        double* get_TE_ptr()
        double* get_TE_pval_ptr()
        double* get_CTE_ptr()
        double* get_CTE_pval_ptr()
        double* get_covariance_ptr()
        double* get_beta_ineff_ptr()

        # These help to find the size of the matrices
        int get_num_markets()
        int get_num_results()














