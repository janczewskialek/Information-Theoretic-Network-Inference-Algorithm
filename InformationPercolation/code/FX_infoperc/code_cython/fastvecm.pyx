# distutils: language = c++

"""
Cython code that wraps the C++ classes in Python.
"""


from VECM cimport VECM
from DailyVECM cimport DailyVECM
from FX_infoperc.pyvecm import VECM_result
import numpy as np
cimport numpy as np


cdef class fastVECM:
    cdef VECM c_vecm
    cdef int num_markets
    cdef int rows, columns
    cdef int lag_num_VECM, lag_num_IRF
    cdef int TE_hist, TE_tau, TE_permutation, TE_auto


    def __cinit__(self,
                  np.ndarray[np.double_t, ndim=2] data,
                  int lag_num_VECM,
                  int lag_num_IRF,
                  int min_obs,
                  int TE_hist,
                  int TE_tau,
                  int TE_permutation,
                  int TE_auto):

        """Initialize the class instance.

        Args:
            data (numpy.ndarray): data in a numpy ndarray, each column represens a variable
            lag_num_VECM (int): the number of lags in the vector error correction model
            lag_num_IRF (int): the number of lags in the VMA model
            min_obs (int): the minimum number of observations

        Returns: -

        """
        # Size of the input array
        cdef int num_rows = data.shape[0]
        cdef int num_cols = data.shape[1]

        # Specify class variables
        self.num_markets = num_cols
        self.rows = num_rows
        self.columns = num_cols
        self.lag_num_VECM = lag_num_VECM
        self.lag_num_IRF = lag_num_IRF
        self.TE_hist        = TE_hist
        self.TE_tau         = TE_tau
        self.TE_permutation = TE_permutation
        self.TE_auto        = TE_auto


        # Data must be continuous
        data = np.require(data, np.double, ['F', 'A'])

        cdef double[:, :] data_memview = data
        self.c_vecm = VECM(&data_memview[0, 0], num_rows, num_cols, lag_num_VECM, lag_num_IRF, min_obs, TE_hist, TE_tau, TE_permutation, TE_auto)


    def run_models(self, int matrix_transform=1):
        """Fit the vector error correction model, and returns with the results
        in a VECM_result object.

        Args:
            matrix_transform (int): how to compute the long-term impacts
                                    0: use matrix transformation (less stable, faster)
                                    1: use the VMA approach

        Returns (VECM_result): the result of the analysis, includeing
                                    S_lower: lower bound of information share
                                    S_upper: upper boubnd of information share
                                    long_run: long-term impacts
                                    component_share: component share
                                    edges: information flows (sensitive to variance)
                                    edges_corr: information flows with correlation (not sensitive to variance)
                                    covariance: covariance matrix of disturbances
                                    included: whether a market is included of not (e.g. number of price changes > 50)
        """
        self.c_vecm.run_models(matrix_transform)
        res = VECM_result(S_lower=self.S_lower,
                          S_upper=self.S_upper,
                          long_run=self.long_run,
                          component_share=self.component_share,
                          edges=self.edges,
                          edges_corr=self.edges_corr,
                          covariance=self.covariance,
                          included=self.included,
                          beta_inefficiency = self.beta_inefficiency,
                          TE = self.TE,
                          TE_pval = self.TE_pval,
                          CTE = self.CTE,
                          CTE_pval = self.CTE_pval)
        return res

    @property
    def S_lower(self):
        cdef double* ptr = self.c_vecm.get_S_lower_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((1, self.num_markets), order='F')

    @property
    def S_upper(self):
        cdef double* ptr = self.c_vecm.get_S_upper_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((1, self.num_markets), order='F')

    @property
    def long_run(self):
        cdef double* ptr = self.c_vecm.get_long_run_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((1, self.num_markets), order='F')

    @property
    def component_share(self):
        cdef double* ptr = self.c_vecm.get_component_share_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((1, self.num_markets), order='F')

    @property
    def beta_inefficiency(self):
        cdef double* ptr = self.c_vecm.get_beta_ineff_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_markets*self.lag_num_IRF]>ptr, dtype=np.double)
        return np_arr.reshape((self.lag_num_IRF, self.num_markets), order='F')

    @property
    def edges(self):
        cdef double* ptr = self.c_vecm.get_edges_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')


    @property
    def TE(self):
        cdef double* ptr = self.c_vecm.get_TE_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')
    
    @property
    def TE_pval(self):
        cdef double* ptr = self.c_vecm.get_TE_pval_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')

    @property
    def CTE(self):
        cdef double* ptr = self.c_vecm.get_CTE_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')

    @property
    def CTE_pval(self):
        cdef double* ptr = self.c_vecm.get_CTE_pval_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')



    @property
    def edges_corr(self):
        cdef double* ptr = self.c_vecm.get_edges_corr_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2 * self.lag_num_IRF)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets * self.lag_num_IRF, self.num_markets), order='F')

    @property
    def covariance(self):
        cdef double* ptr = self.c_vecm.get_covariance_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:int(self.num_markets**2)]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_markets, self.num_markets), order='F')

    @property
    def data(self):
        cdef double* ptr = self.c_vecm.get_data_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.rows*self.columns]>ptr, dtype=np.double)
        return np_arr.reshape((self.rows, self.columns), order='F')

    @property
    def included(self):
        cdef int* ptr = self.c_vecm.get_included_ptr()
        cdef np.ndarray np_arr = np.asarray(<int[:self.num_markets]>ptr, dtype=int)
        return np_arr.reshape((1, self.columns), order='F')

    @property
    def alpha(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef double* ptr = self.c_vecm.get_alpha_ptr()
        cdef np.ndarray np_arr = np.asarray(<double[:num_included*(num_included-1)]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, num_included-1), order='F')

    @property
    def pi(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef double* ptr = self.c_vecm.get_pi_ptr()
        cdef np.ndarray np_arr = np.asarray(<double[:num_included*num_included]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, num_included), order='F')

    @property
    def beta(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef double* ptr = self.c_vecm.get_beta_ptr()
        cdef np.ndarray np_arr = np.asarray(<double[:num_included*(num_included-1)]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, num_included-1), order='F')

    @property
    def gamma(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef double* ptr = self.c_vecm.get_gamma_ptr()
        cdef np.ndarray np_arr = np.asarray(<double[:num_included*num_included*self.lag_num_VECM]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, self.lag_num_VECM*num_included), order='F')

    @property
    def theta(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef double* ptr = self.c_vecm.get_theta_ptr()
        cdef np.ndarray np_arr = np.asarray(<double[:num_included*num_included*self.lag_num_IRF]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, self.lag_num_IRF*num_included), order='F')

    def print_all(self):
        self.c_vecm.print_all()

    @property
    def dZ(self):
        cdef int num_dZ = self.c_vecm.get_dZ_num()

        cdef double* ptr = self.c_vecm.get_dZ_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:num_dZ]>ptr, dtype=np.double)
        return np_arr.reshape((1, num_dZ), order='F')[0]

    @property
    def residuals(self): ### only for the included markets
        cdef int num_included = self.c_vecm.get_num_included()
        cdef int num_dZ = self.c_vecm.get_dZ_num()

        # Get pointer to the array
        cdef double* ptr = self.c_vecm.get_residuals_ptr()

        cdef np.ndarray np_arr = np.asarray(<double[:num_included*num_dZ]>ptr, dtype=np.double)
        return np_arr.reshape((num_included, num_dZ), order='F')


cdef class fastDailyVECM:

    cdef DailyVECM c_dailyvecm
    cdef int num_markets
    cdef int num_results
    cdef int lag_num_IRF


    def __cinit__(self,
                    int[:] time,
                    int[:] market_id,
                    double[:] ask,
                    double[:] bid):

        cdef int n = np.max([time.shape[0], time.shape[1]])

        #print("Size", n)

        self.c_dailyvecm = DailyVECM(&time[0],
                                     &market_id[0],
                                     &ask[0],
                                     &bid[0],
                                     n)


    def init(self, int fs_ms, int window_size_ms, int step_size_ms, int begin_ms, int end_ms):
        """
        Initialize the object.

        Args:
            fs_ms (int): sampling period in ms, e.g. 25
            window_size_ms (int): the length of the sliding window in ms
            step_size_ms (int): the step size of the sliding window in ms
            begin_ms (int): the beginning of the period in ms, min 0.
            end_ms (int): the end of the period in ms, max. 26*60*60*1000

        Important:
            begin_ms < end_ms

        Returns: -

        """
        self.c_dailyvecm.init(fs_ms, window_size_ms, step_size_ms, begin_ms, end_ms)

    def run(self,
            int num_threads,
            int lag_num_VECM,
            int lag_num_IRF,
            int min_obs,
            int matrix_transform,
            int TE_hist,
            int TE_tau,
            int TE_permutation,
            int TE_auto):
        """Run the analysis within the specified time period.

        Args:
            num_threads (int): number of threads to use
            lag_num_VECM (int): number of lags in the VECM model
            lag_num_IRF (int): number of lags in the VMA model
            min_obs (int): minimum number of observations
            matrix_transform (int): how to compute the long-term impacts.
                                    1: VMA (stable, slower)
                                    0: matrix transform (less stable, faster)

        Returns: -

        """

        self.c_dailyvecm.run(num_threads, lag_num_VECM, lag_num_IRF, min_obs, matrix_transform, TE_hist,TE_tau, TE_permutation,TE_auto)

        self.num_markets = self.c_dailyvecm.get_num_markets()
        self.num_results = self.c_dailyvecm.get_num_results()
        self.lag_num_IRF = lag_num_IRF
        #print("Number of markets", self.num_markets)
        #print("Number of results", self.num_results)

    @property
    def time(self):
        cdef double* ptr = self.c_dailyvecm.get_time_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results]>ptr, dtype=np.double)
        return np_arr

    @property
    def included(self):
        cdef int* ptr = self.c_dailyvecm.get_included_ptr()
        cdef np.ndarray np_arr = np.asarray(<int[:self.num_results*self.num_markets]>ptr, dtype=int)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def spread(self):
        cdef double* ptr = self.c_dailyvecm.get_spread_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def variance(self):
        cdef double* ptr = self.c_dailyvecm.get_variance_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def S_lower(self):
        cdef double* ptr = self.c_dailyvecm.get_S_lower_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def S_upper(self):
        cdef double* ptr = self.c_dailyvecm.get_S_upper_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def long_run(self):
        cdef double* ptr = self.c_dailyvecm.get_long_run_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def component_share(self):
        cdef double* ptr = self.c_dailyvecm.get_component_share_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets), order='F')

    @property
    def beta_inefficiency(self):
        cdef double* ptr = self.c_dailyvecm.get_beta_ineff_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets *self.lag_num_IRF]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets *self.lag_num_IRF), order='F')

    @property
    def edges(self):
        cdef double* ptr = self.c_dailyvecm.get_edges_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')



    @property
    def TE(self):
        cdef double* ptr = self.c_dailyvecm.get_TE_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')

    @property
    def TE_pval(self):
        cdef double* ptr = self.c_dailyvecm.get_TE_pval_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')

    @property
    def CTE(self):
        cdef double* ptr = self.c_dailyvecm.get_CTE_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')

    @property
    def CTE_pval(self):
        cdef double* ptr = self.c_dailyvecm.get_CTE_pval_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')





    @property
    def edges_corr(self):
        cdef double* ptr = self.c_dailyvecm.get_edges_corr_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets*self.lag_num_IRF]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2 * self.lag_num_IRF), order='F')

    @property
    def covariance(self):
        cdef double* ptr = self.c_dailyvecm.get_covariance_ptr()
        cdef np.ndarray np_arr = np.asarray(<np.double_t[:self.num_results*self.num_markets*self.num_markets]>ptr, dtype=np.double)
        return np_arr.reshape((self.num_results, self.num_markets**2), order='F')



"""
            double* get_S_lower_ptr()
        double* get_S_upper_ptr()
        double* get_long_run_ptr()
        double* get_component_share_ptr()
        double* get_edges_ptr()
        double* get_covariance_ptr()

        #numpy_array = np.asarray(<np.int32_t[:10, :10]> my_pointer)
"""


