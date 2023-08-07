#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Cholesky"
#include <iostream>
#include <string>
#include <tuple>
#include <chrono>
#include <iomanip>
#include <stdio.h>

#ifndef C_PROJECT_VECM_H
#define C_PROJECT_VECM_H

/**
 * @brief Class that implements the information revelation model by Hagströmer and Menkveld (2019) and encapsulates the Information Theoretic Network Inference Algorithm.
 *
 * This class implements the information revelation model developed by Hagströmer and Menkveld (2019) and encapsulates the Information Theoretic Network Inference Algorithm.
 *
 * @author Aleksander Janczewski
 * @date August 2023
 */


using namespace Eigen;

class VECM {
    /** Vector error correction model parameters */
    int min_obs; /**< Minimum number of price changes to include a market */
    int lag_num_VECM; /**< Number of lags in the VEC model */
    int lag_num_IRF; /**< Number of lags in the Impulse Response Function */

    Matrix<bool, Dynamic, Dynamic> included; /**< Indicate whether a market is included */
    MatrixXi included_int; /**< Indicate whether a market is included in the sample*/
    MatrixXi num_changes; /** < Number of price changes for each market */
    int num_original; /**< The number of markets in the input dataset */
    int num_included; /**< The number of market that have enough price changes */

    /** Information-theoretic network inference algorithm parameters */
    int TE_hist;  /**< Maximum embedding dimension for both source and target (if TE_auto = 1), else the embedding dimension to be used for both source and target */
    int TE_tau; /**< Maximum embedding delay for both source and target (if TE_auto = 1), else the embedding delay to be used for both source and target */
    int TE_permutation; /**< Number of permutations to performed to determine the p-value of the estimate */
    int TE_auto; /**< Boolean whether optimal dimension and delay embedding should be performed using Ragwitz's criterion (1-yes, 0-no) */


    /** Containers */
    MatrixXd data_input, data_input_diff, data, data_diff;
    MatrixXd dY, Y_1, dX, residuals;
    MatrixXd alpha, beta, pi, gamma;
    MatrixXd theta;
    MatrixXd dZ;
    MatrixXd S_lower, S_upper, long_run, component_share, edges, edges_corr, covariance, beta_ineff;
    MatrixXd S_lower_full, S_upper_full, long_run_full, component_share_full, edges_full, edges_corr_full, covariance_full, beta_ineff_full;

    /** Containers for Information Theoretic Network Inference Algorithm results */
    MatrixXd TE, TE_pval, CTE, CTE_pval;
    MatrixXd TE_full, TE_pval_full, CTE_full, CTE_pval_full;


    /** @brief
     *
     * Fit a vector error correction model to the data with a least squares approach. Check Lütkepohl (2008) for details.
     *
     * @param dY first difference of prices
     * @param Y_1 price levels
     * @param dX first difference of prices but with one lag
     * @param beta cointegration matrix
     * @param lag_num_VECM number of lags in the vector error correction model
     * */
    std::tuple<MatrixXd, MatrixXd>
    fit_vecm_ls(MatrixXd &dY, MatrixXd &Y_1, MatrixXd &dX, MatrixXd &beta, int lag_num_VECM);


    /** @brief Determine lag selection based on BIC criterion. Check Lütkepohl (2008) for details.
    * */
    void bic_lag_selection(int &lag_num_VECM, MatrixXd &dY, MatrixXd &Y_1, MatrixXd &beta);

    /** @brief Fit a Impulse Response Function to the disturbances to compute the long-term impacts based on Hagstromer and Menkveld (2019) implementation.
     * */
    std::tuple<MatrixXd, MatrixXd> fit_IRF(MatrixXd &dY, MatrixXd &alpha, MatrixXd &beta,
                                           MatrixXd &gamma, int lag_num_IRF, int lag_num_VECM);

    /** @brief Computes information shares from Joel Hasbrouck (1995) */
    std::tuple<MatrixXd, MatrixXd> calc_information_share(MatrixXd &long_run, MatrixXd &covariance);

    /** @brief Computes edges based on partial correlations analogous to Hagstromer and Menkveld (2019) implementation */
    std::tuple<MatrixXd, MatrixXd> calc_edges(MatrixXd &dY, MatrixXd &dZ,
                                              MatrixXd &theta, MatrixXd &covariance,
                                              int lag_num_IRF);

    /** @brief Computes beta inefficientcy (Hagstromer and Menkveld (2019) */
    MatrixXd calc_beta_ineff(MatrixXd &theta, MatrixXd &covariance, MatrixXd &long_run, int lag_num_IRF);

    /** @brief Print a (double) matrix to std::cout */
    void print_matrix(MatrixXd mat, std::string text, int mode = 0);

    /** @brief Print a matrix (int) to std::cout */
    void print_matrix(MatrixXi mat, std::string text, int mode = 0);

    /** @brief Print a matrix (bool) to std::cout */
    void print_matrix(Matrix<bool, Dynamic, Dynamic> mat, std::string text, int mode = 0);

    /** @brief Extends a vector to include all markets */
    MatrixXd extend_vector(MatrixXd &vector, Matrix<bool, Dynamic, Dynamic> &included);

    /** @brief Extends a matrix to include all markets */
    MatrixXd extend_matrix(MatrixXd &vector, Matrix<bool, Dynamic, Dynamic> &included);

    /** @brief Extends a matrix to include all markets for INA results, not included stand as NAN*/
    MatrixXd extend_matrix_INA(MatrixXd &vector, Matrix<bool, Dynamic, Dynamic> &included);

    /** @brief Extends a matrix with VMAlags number of rows to include all markets */
    MatrixXd extend_matrix_lag(MatrixXd &vector, Matrix<bool, Dynamic, Dynamic> &included, int lag_num_IRF, bool PI);

public:
    /** Constructor */
    VECM();

    /** Constructor */
    VECM(double *data,  // Data
         int num_rows,
         int num_cols,
         int lag_num_VECM, // VECM parameters
         int lag_num_IRF,
         int min_obs,
         int TE_hist,
         int TE_tau,
         int TE_permutation,
         int TE_auto);

    /** Destructor */
    virtual ~VECM();

    /** Run the analysis
     * @param run_INA should run information-theoretic network inference algorithm?
     * */
    void run_models(int run_INA);


    /** Print all matrices */
    void print_all();

    // Interface to Python
    int get_num_included(); /**< Get the number of included markets */
    double *get_data_ptr(); /**< Get a pointer to the input dataset */
    double *get_S_lower_ptr(); /**< Get a pointer to a vector containing the lower bound of information shares */
    double *get_S_upper_ptr(); /**< Get a pointer to a vector containing the upper bounds of information shares */
    double *get_long_run_ptr(); /**< Get a pointer to a vector containing the long-term impacts */
    double *get_component_share_ptr(); /**< Get a pointer to a vector containing the component share */
    double *get_edges_ptr(); /**< Get a pointer to a matrix containing the edges */
    double *get_edges_corr_ptr(); /**< Get a pointer to a matrix containing the edges computed with correlation */
    double *get_covariance_ptr(); /**< Get a pointer to the covariance matrix of disturbances */
    int *get_included_ptr(); /**< Get a pointer to a vector indicating which matrix is included */
    double *get_beta_ineff_ptr(); /**< Get a pointer to a vector of beta inefficiencies */
    double *get_residuals_ptr(); /**< Get a pointer to the matrix of residuals */

    double *get_alpha_ptr(); /**< Get a pointer to the alpha (loading matrix) matrix */
    double *get_pi_ptr(); /**< Get a pointer to the pi matrix */
    double *get_beta_ptr(); /**<  Get a pointer to the beta (cointegrating matrix) matrix*/
    double *get_gamma_ptr(); /**< Get a pointer to the gamma matrix (VECM coefficients) */
    double *get_theta_ptr(); /**< Get a pointer to the theta matrix (only for the VMA model) */
    double *get_dZ_ptr(); /**< Get a pointer to the estimated efficient price innovations */
    int get_dZ_num(); /**< Length of the dZ vector */

    double *get_TE_ptr(); /**< Get a pointer to the Transfer Entropy estimates matrix */
    double *get_TE_pval_ptr(); /**< Get a pointer to the Transfer Entropy pvalues matrix */
    double *get_CTE_ptr(); /**< Get a pointer to the Conditional Transfer Entropy results matrix */
    double *get_CTE_pval_ptr(); /**< Get a pointer to the Conditional Transfer Entropy results matrix */

    // Interface to other C++ applications
    MatrixXd get_S_lower(); /**< Get the lower bound of information shares */
    MatrixXd get_S_upper(); /**< Get the upper bound of information shares */
    MatrixXd get_long_run(); /**< Get the long-term impacts */
    MatrixXd get_component_share(); /**< Get the component shares */
    MatrixXd get_edges(); /**< Get the matrix of information flows */
    MatrixXd get_edges_corr(); /**< Get the matrix of information flows computed with correlation */
    MatrixXd get_covariance(); /**< Get the covariance matrix of estimated disturbances */
    MatrixXi get_included(); /**< Get a vector indicating the included markets */
    MatrixXd get_beta_ineff(); /**< Get a vector of beta inefficiencies */

    MatrixXd get_alpha(); /**< Get the alpha matrix */
    MatrixXd get_pi(); /**< Get the pi matrix */
    MatrixXd get_beta(); /**< Get the beta matrix */
    MatrixXd get_gamma(); /**< Get the gamma matrix */
    MatrixXd get_theta(); /**< Get the theta matrix (VMA model) */
    MatrixXd get_residuals(); /**< Get the matrix of estimated residuals */
    MatrixXd get_dZ(); /**< Get the vector of estimated efficient price innovations */

    MatrixXd get_TE(); /**< Get a pointer to the Transfer Entropy estimates matrix */
    MatrixXd get_TE_pval(); /**< Get a pointer to the Transfer Entropy pvalues matrix */
    MatrixXd get_CTE(); /**< Get a pointer to the Transfer Entropy results matrix */
    MatrixXd get_CTE_pval(); /**< Get a pointer to the Transfer Entropy results matrix */

};


#endif //C_PROJECT_VECM_H
