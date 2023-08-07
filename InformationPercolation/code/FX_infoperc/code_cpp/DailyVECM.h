#include "Eigen/Core"
#include "Eigen/Dense"

#ifndef PR1_DAILYVECM_H
#define PR1_DAILYVECM_H

/**
 * @brief Implements a parallelization of VECM class with sliding window that can run through a period
 *
 * @author Aleksander Janczewski
 * @date August 2023
 */


using namespace Eigen;

class DailyVECM {

    // Input parameters
    int *time;
    int* market_id;
    double* ask;
    double* bid;
    int n;
    int num_markets;
    int fs_ms;
    int step_size_num;
    int window_length_num;


    MatrixXd mid, spread;
    MatrixXi time_vector;

    /** Matrices that store the results
     * Each row represents a time window.
     * */
    MatrixXd res_time;
    MatrixXi res_included;
    MatrixXd res_spread;
    MatrixXd res_variance;
    MatrixXd res_S_lower;
    MatrixXd res_S_upper;
    MatrixXd res_long_run;
    MatrixXd res_component_share;
    MatrixXd res_edges;
    MatrixXd res_edges_corr;
    MatrixXd res_covariance;
    MatrixXd res_beta_ineff;

    MatrixXd res_TE;
    MatrixXd res_TE_pval;
    MatrixXd res_CTE;
    MatrixXd res_CTE_pval;

    /**
     * @brief This function runs parallel and analyses one time window
     *
     * */
    void worker(int id,
                int num_workers,
                MatrixXd& mid,
                MatrixXd& spread,
                MatrixXi& time_vector,
                MatrixXi& start_index,
                int window_length_num,
                MatrixXd& res_time,
                MatrixXd& res_spread,
                MatrixXd& res_variance,
                MatrixXd& res_S_lower,
                MatrixXd& res_S_upper,
                MatrixXd& res_long_run,
                MatrixXd& res_component_share,
                MatrixXd& res_edges,
                MatrixXd& res_edges_corr,
                MatrixXd& res_covariance,
                MatrixXd& res_beta_ineff,
                MatrixXi& included,
                MatrixXd& res_TE,
                MatrixXd& res_TE_pval,
                MatrixXd& res_CTE,
                MatrixXd& res_CTE_pval,
                int lag_num_VECM,
                int lag_num_IRF,
                int min_obs,
                int matrix_transform,
                int TE_hist,
                int TE_tau,
                int TE_permutation,
                int TE_auto);

public:
    /** Constructor */
    DailyVECM();
    /** Constructor */
    DailyVECM(int *time, int* market_id, double* ask, double* bid, int n);
    /** Init, set the internal matrices */
    void init(int fs_ms, int window_size_ms, int step_size_ms, int begin_ms, int end_ms);
    /** Launch multiple threads */
    void run(int num_threads,
            int lag_num_VECM,
            int lag_num_IRF,
            int min_obs,
            int matrix_transform,
            int TE_hist,
            int TE_tau,
            int TE_permutation,
            int TE_auto);


    /** Get pointers to the matices, so that we can read them from Python. */
    double* get_time_ptr();
    int* get_included_ptr();
    double* get_spread_ptr();
    double* get_variance_ptr();
    double* get_S_lower_ptr();
    double* get_S_upper_ptr();
    double* get_long_run_ptr();
    double* get_component_share_ptr();
    double* get_edges_ptr();
    double* get_edges_corr_ptr();
    double* get_TE_ptr();
    double* get_TE_pval_ptr();
    double* get_CTE_ptr();
    double* get_CTE_pval_ptr();
    double* get_covariance_ptr();
    double* get_beta_ineff_ptr();
    int get_num_markets();
    int get_num_results();

    int VMA_dim;
};


#endif //PR1_DAILYVECM_H
