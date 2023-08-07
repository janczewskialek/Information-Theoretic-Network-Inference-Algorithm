#include "VECM.h"
#include <cmath>
#include "unsupported/Eigen/MatrixFunctions"
#include "Eigen/Eigenvalues"
#include "../../../../TransferEntropy/INA.h"

VECM::~VECM() = default;

VECM::VECM() = default;

VECM::VECM(double *data,
           int num_rows,
           int num_cols,
           int lag_num_VECM,
           int lag_num_IRF,
           int min_obs,
           int TE_hist,
           int TE_tau,
           int TE_permutation,
           int TE_auto) {

    this->num_original = num_cols;
    this->lag_num_VECM = lag_num_VECM;
    this->lag_num_IRF = lag_num_IRF;
    this->min_obs = min_obs;

    this->TE_hist = TE_hist;
    this->TE_tau = TE_tau;
    this->TE_permutation = TE_permutation;
    this->TE_auto = TE_auto;


    // Convert data to Eigen matrix
    this->data_input = MatrixXd::Map(data, num_rows, num_cols);
    this->data_input_diff = data_input.bottomRows(num_rows - 1) - data_input.topRows(num_rows - 1);

    // Count the number of changes for each time series
    this->num_changes = (this->data_input_diff.array() != 0).colwise().count().cast<int>();

    // Determine which market can be included given the minimum observations conditiopn
    this->included = (this->num_changes.array() >= min_obs);

    // Fill included with int (convenient to return with this)
    this->included_int = MatrixXi::Zero(1, this->num_original);

    for (int i = 0; i < this->num_original; i = i + 1) {
        if (this->included(i)) {
            this->included_int(i) = 1;
        } else {
            this->included_int(i) = 0;
        }
    }

    this->num_included = (int) this->included.count();

    // make sure at least 2 time series included in the run
    if (this->num_included >= 2) {
        // Copy data to the other container
        this->data = MatrixXd(this->data_input.rows(), this->num_included);
        this->data_diff = MatrixXd(this->data_input_diff.rows(), this->num_included);
        for (int i = 0, j = 0; i < this->num_original; i = i + 1) {
            if (this->included(i)) {
                this->data.col(j) = this->data_input.col(i);
                this->data_diff.col(j) = this->data_input_diff.col(i);
                j = j + 1;
            }
        }

        // Compute dY
        this->dY = this->data_diff.transpose();
        this->Y_1 = this->data.topRows(this->data.rows() - 1).transpose();

        // Beta setup per Menkveld and Hastromer 2019 (aligns with the results)
        this->beta = MatrixXd::Identity(this->num_included, this->num_included);

        // BIC lag selection, changes lag_num_VECM by reference
        bic_lag_selection(this->lag_num_VECM, this->dY, this->Y_1, this->beta);

        // dX adjusted with respect to num_included
        this->dX = MatrixXd(this->lag_num_VECM * this->num_included, this->dY.cols());
        this->dX.array().block(0, 0, this->dX.rows(), this->lag_num_VECM + 1) = 0;
        for (int i = 0; i < this->lag_num_VECM; i = i + 1) {
            this->dX.block(i * this->num_included, i + 1, this->num_included, this->dY.cols() - i - 1) = this->dY.block(
                    0, 0, this->num_included, this->dY.cols() - i - 1);
        }

    }


    // Initialize output containers
    this->S_lower_full = MatrixXd::Zero(1, this->num_original);
    this->S_upper_full = MatrixXd::Zero(1, this->num_original);
    this->long_run_full = MatrixXd::Zero(1, this->num_original);
    this->component_share_full = MatrixXd::Zero(1, this->num_original);
    this->edges_full = MatrixXd::Zero(this->num_original, this->lag_num_IRF);
    this->edges_corr_full = MatrixXd::Zero(this->num_original * lag_num_IRF, this->num_original);
    this->covariance_full = MatrixXd::Zero(this->num_original, this->num_original);
    this->beta_ineff_full = MatrixXd::Zero(this->lag_num_IRF, this->num_original);

    //Containers for INA
    this->TE_full = MatrixXd::Zero(this->num_original, this->num_original);
    this->TE_pval_full = MatrixXd::Zero(this->num_original, this->num_original);
    this->CTE_full = MatrixXd::Zero(this->num_original, this->num_original);
    this->CTE_pval_full = MatrixXd::Zero(this->num_original, this->num_original);


}

//----------------------MATRIX PRINTING FUNCTIONS----------------------
void VECM::print_matrix(MatrixXd mat, std::string text, int mode) {
    std::cout << "\n-------------------------\n";
    std::cout << text << "\t(" << mat.rows() << ", " << mat.cols() << ")\n";
    if (mode == 1) {
        std::cout << mat << std::endl;
    }
    std::cout << "\n-------------------------\n";
}

void VECM::print_matrix(MatrixXi mat, std::string text, int mode) {
    std::cout << "\n-------------------------\n";
    std::cout << text << "\t(" << mat.rows() << ", " << mat.cols() << ")\n";
    if (mode == 1) {
        std::cout << mat << std::endl;
    }
    std::cout << "\n-------------------------\n";
}

void VECM::print_matrix(Matrix<bool, Dynamic, Dynamic> mat, std::string text, int mode) {
    std::cout << "\n-------------------------\n";
    std::cout << text << "\t(" << mat.rows() << ", " << mat.cols() << ")\n";
    if (mode == 1) {
        std::cout << mat << std::endl;
    }
    std::cout << "\n-------------------------\n";
}


void VECM::bic_lag_selection(int &lag_num_VECM, MatrixXd &dY, MatrixXd &Y_1, MatrixXd &beta) {

    int lag_num_VECM_MAX = lag_num_VECM;

    MatrixXd pi;
    MatrixXd BIC = MatrixXd(1, lag_num_VECM_MAX);
    MatrixXd dX_ls;
    MatrixXd dY_ls = dY;
    MatrixXd Y_1_ls = Y_1;
    MatrixXd beta_ls = beta;
    MatrixXd alpha_ls;
    MatrixXd gamma_ls;
    MatrixXd residuals_ls;
    MatrixXd residuals_ls_SQRS;
    MatrixXd sse;
    MatrixXd omega;
    MatrixXd sigma_u_mle;

    double lowest_BIC;
    double nobs = dY_ls.cols() + 1;
    double df_resid;
    double free_params;
    double ld;

    int num_included_ls = num_included;

    for (int i = 1; i < lag_num_VECM_MAX + 1; i = i + 1) {
        // reset ld for each iteration of lag_num
        ld = 0;

        dX_ls = MatrixXd(i * num_included_ls, dY_ls.cols());
        dX_ls.array().block(0, 0, dX_ls.rows(), i + 1) = 0;
        for (int j = 0; j < i; j = j + 1) {
            dX_ls.block(j * num_included_ls, j + 1, num_included_ls, dY_ls.cols() - j - 1)
                    = dY_ls.block(0, 0, num_included_ls, dY_ls.cols() - j - 1);
        }

        std::tuple<MatrixXd, MatrixXd> tuple_alpha_gamma = fit_vecm_ls(dY_ls, Y_1_ls, dX_ls, beta_ls, i);
        alpha_ls = std::get<0>(tuple_alpha_gamma);
        gamma_ls = std::get<1>(tuple_alpha_gamma);

        pi = alpha_ls * beta_ls.transpose();
        residuals_ls = dY_ls - (pi * Y_1_ls + gamma_ls * dX_ls);

        df_resid = nobs - (num_included_ls * i + 1);
        sse = residuals_ls * residuals_ls.transpose();
        omega = sse / df_resid;
        free_params = i * num_included_ls * num_included_ls;
        sigma_u_mle = omega * df_resid / nobs;

        MatrixXd U = sigma_u_mle.llt().matrixL();
        for (int k = 0; k < sigma_u_mle.rows(); k = k + 1)
            ld += log(U(k, k));
        ld *= 2;

        BIC(0, i - 1) = ld + log(nobs) / nobs * free_params;
    }

    lowest_BIC = BIC.minCoeff();

    for (int i = 0; i < BIC.cols(); i = i + 1) {
        if (lowest_BIC == BIC(0, i)) {
            lag_num_VECM = i + 1;  // intentionally +1 since the index of the array starts with 0 associated with lag 1
        }
    }

}

void VECM::run_models(int run_INA) {

    switch (this->num_included) {
        case 0:
            // Do not modify containers
            break;
        case 1: {
            // Determine which market is included
            int index = 0;
            for (int i = 0; i < this->num_original; i = i + 1) {
                if (this->included(i)) {
                    index = i;
                }
            }
            this->S_lower_full(0, index) = 1;
            this->S_upper_full(0, index) = 1;
            this->long_run_full(0, index) = 1;
            this->component_share_full(0, index) = 1;
            this->beta_ineff_full(0, index) = 1;

            break;
        }
        default: {
            // Fit VEC model parameters
            std::tuple<MatrixXd, MatrixXd> tuple_alpha_gamma = this->fit_vecm_ls(this->dY, this->Y_1, this->dX,
                                                                                 this->beta, this->lag_num_VECM);
            this->alpha = std::get<0>(tuple_alpha_gamma);
            this->gamma = std::get<1>(tuple_alpha_gamma);

            // Calculate residuals (residuals = observation - prediction)
            MatrixXd pi = this->alpha * this->beta.transpose();
            this->pi = pi;
            this->residuals.noalias() = this->dY - (pi * this->Y_1 + this->gamma * this->dX);

            std::tuple<MatrixXd, MatrixXd> tuple_long_run_theta = this->fit_IRF(this->dY, this->alpha, this->beta,
                                                                                this->gamma, this->lag_num_IRF,
                                                                                this->lag_num_VECM);

            this->long_run = std::get<0>(tuple_long_run_theta);
            this->theta = std::get<1>(tuple_long_run_theta);

            // Compute component share
            if (fabs(this->long_run.sum()) > 1.0e-10) {
                this->component_share = this->long_run.array() / this->long_run.sum();
            } else {
                this->component_share = this->long_run;
            }

            // Compute covariance matrix
            MatrixXd centered = this->residuals.transpose().rowwise() - this->residuals.transpose().colwise().mean();
            this->covariance.noalias() =
                    (centered.adjoint() * centered) / double(this->residuals.transpose().rows() - 1);


            // Compute lower and upper information share
            std::tuple<MatrixXd, MatrixXd> tuple_S_lower_S_upper = this->calc_information_share(this->long_run,
                                                                                                this->covariance);
            this->S_lower = std::get<0>(tuple_S_lower_S_upper);
            this->S_upper = std::get<1>(tuple_S_lower_S_upper);

            // Efficient price increments
            this->dZ.noalias() = this->long_run * this->residuals;

            // Determine edges
            std::tuple<MatrixXd, MatrixXd> tuple_edges = this->calc_edges(this->dY, this->dZ,
                                                                          this->theta, this->covariance,
                                                                          this->lag_num_IRF);

            // Edges not used, edges_corr is used
            this->edges = std::get<0>(tuple_edges);
            this->edges_corr = std::get<1>(tuple_edges);

            // Price inefficiency
            this->beta_ineff = this->calc_beta_ineff(this->theta, this->covariance, this->long_run, this->lag_num_IRF);

            // Information-theoretic Network Analysis
            if (run_INA == true) {

                bool AUTOEMB = (TE_auto) ? 1 : 0;
                bool AUTODEL_TE = (TE_auto) ? 1 : 0;
                bool AUTODEL_CTE = (TE_auto) ? 1 : 0;
                std::string is_automatic= (TE_auto) ? "TE_auto is ON" : "TE_auto is OFF";

                std::cout << is_automatic << std::endl;

                int MAX_HIST = this->TE_hist;
                int MAX_TAU = this->TE_tau;
                int X_hist = this->TE_hist;
                int Y_hist = this->TE_hist;
                int X_tau = this->TE_tau;
                int Y_tau = this->TE_tau;
                int permutation = this->TE_permutation;

                // NOTE!: The following parameters are hardcoded for the Python wrapped library!
                bool CTE_analysis = 1;
                int MAX_DELAY = 4;
                int X_delay = 1;
                double min_pval_TE = 0.05;
                double min_pval_CTE = 0.05;
                bool STD = 1;
                int NOISE = 12;
                int k = 4;
                int leaves = 100;
                bool DEBUG = 0;


                auto [TE_results, TE_pval_results, CTE_results, CTE_pval_results]
                        = Information_Network_Analysis(this->dY,
                                                       CTE_analysis,
                                                       AUTOEMB,
                                                       MAX_HIST,
                                                       MAX_TAU,
                                                       AUTODEL_TE,
                                                       AUTODEL_CTE,
                                                       MAX_DELAY,
                                                       X_hist,
                                                       Y_hist,
                                                       X_tau,
                                                       Y_tau,
                                                       X_delay,
                                                       permutation,
                                                       min_pval_TE,
                                                       min_pval_CTE,
                                                       STD,
                                                       NOISE,
                                                       k,
                                                       leaves,
                                                       DEBUG);

                this->TE = TE_results;
                this->TE_pval = TE_pval_results;
                this->CTE = CTE_results;
                this->CTE_pval = CTE_pval_results;
            }

            // Extend vectors, if less markets have been included than received
            this->S_lower_full = this->extend_vector(this->S_lower, this->included);
            this->S_upper_full = this->extend_vector(this->S_upper, this->included);
            this->long_run_full = this->extend_vector(this->long_run, this->included);
            this->component_share_full = this->extend_vector(this->component_share, this->included);
            this->edges_full = this->extend_matrix(this->edges, this->included);
            this->covariance_full = this->extend_matrix(this->covariance, this->included);
            this->edges_corr_full = this->extend_matrix_lag(this->edges_corr, this->included, this->lag_num_IRF, 0);
            this->beta_ineff_full = this->extend_matrix_lag(this->beta_ineff, this->included, this->lag_num_IRF, 1);

            if (run_INA) {
                this->TE_full = this->extend_matrix_INA(this->TE, this->included);
                this->TE_pval_full = this->extend_matrix_INA(this->TE_pval, this->included);
                this->CTE_full = this->extend_matrix_INA(this->CTE, this->included);
                this->CTE_pval_full = this->extend_matrix_INA(this->CTE_pval, this->included);
            }
            break;
        }
    }
}


MatrixXd VECM::calc_beta_ineff(MatrixXd &theta, MatrixXd &covariance, MatrixXd &long_run, int lag_num_IRF) {

    int n = covariance.rows();
    int lags = lag_num_IRF;
    MatrixXd theta_t = theta.transpose();
    MatrixXd lr = long_run;
    MatrixXd omega = covariance;

    MatrixXd beta_ineff = MatrixXd::Zero(n, lags);
    MatrixXd price_ineff = MatrixXd::Zero(n, lags);
    MatrixXd temp_numerator;
    MatrixXd temp_denominator;

    for (int i = 0; i < n * lags; i = i + 1) {
        temp_numerator = theta_t.block(i, 0, 1, n) * omega * lr.transpose();
        temp_denominator = lr * omega * lr.transpose();
        beta_ineff(i % n, i / n) = temp_numerator(0, 0) / temp_denominator(0, 0);
        price_ineff(i % n, i / n) = fabs(1 - beta_ineff(i % n, i / n));
    }
    return price_ineff.transpose();
}

std::tuple<MatrixXd, MatrixXd>
VECM::fit_vecm_ls(MatrixXd &dY, MatrixXd &Y_1, MatrixXd &dX, MatrixXd &beta, int lag_num_VECM) {
    int n = dY.rows();

    MatrixXd a1, a2;
    a1.noalias() = dY * Y_1.transpose() * beta;
    a2.noalias() = dY * dX.transpose();

    MatrixXd a(a1.rows(), a1.cols() + a2.cols());
    a << a1, a2;

    MatrixXd b11, b12, b21, b22;
    b11.noalias() = beta.transpose() * Y_1 * Y_1.transpose() * beta;
    b12.noalias() = beta.transpose() * Y_1 * dX.transpose();
    b21.noalias() = dX * Y_1.transpose() * beta;
    b22.noalias() = dX * dX.transpose();

    MatrixXd b1(b11.rows(), b11.cols() + b12.cols());
    b1 << b11, b12;

    MatrixXd b2(b21.rows(), b21.cols() + b22.cols());
    b2 << b21, b22;

    MatrixXd b(b1.rows() + b2.rows(), b1.cols());
    b << b1, b2;

    MatrixXd b_pinv = b.completeOrthogonalDecomposition().pseudoInverse();
    MatrixXd res = a * b_pinv;
    MatrixXd alpha = res.leftCols(n);
    MatrixXd gamma = res.rightCols(res.cols() - alpha.cols());

    return std::make_tuple(alpha, gamma);
}


std::tuple<MatrixXd, MatrixXd>
VECM::fit_IRF(MatrixXd &dY, MatrixXd &alpha, MatrixXd &beta, MatrixXd &gamma, int lag_num_IRF, int lag_num_VECM) {

    int n = dY.rows();
    int ma = lag_num_VECM;
    int lags = 10000; // note hardcoded number of iterations to be used to find long-run
    double tolerance;

    MatrixXd beta_t = beta.transpose();
    MatrixXd dP = MatrixXd::Zero(n * lags, n * ma); // change in prices vector
    MatrixXd cP = MatrixXd::Zero(n * lags, n); // cumulative change in prices vector
    MatrixXd ccP = MatrixXd::Zero(n * lags, n);
    MatrixXd ccP_t = MatrixXd::Zero(n * lags, n);
    MatrixXd long_run_m = MatrixXd::Zero(1, n);
    MatrixXd dccP_t = MatrixXd::Zero(n, n);
    MatrixXd shocks = MatrixXd::Identity(n, n); // identity matrix with shock for each market
    MatrixXd dY_0 = MatrixXd::Zero(n * ma, n);
    MatrixXd temp;
    MatrixXd temp1;

    // create a matrix for shocks
    dY_0.block(0, 0, n, n) = shocks;
    dY_0.transposeInPlace();

    // iteration over markets
    for (int i = 0; i < n; i = i + 1) {
        // adding a shock for market i assigns the dP for lag 0
        dP.block(i * lags, 0, 1, n * ma) = dY_0.block(i, 0, 1, n * ma);
        cP.block(i * lags, 0, 1, n) = dY_0.block(i, 0, 1, n);

        // iteration over time lags -- starting from lag 1, since lag 0 is assigned in the for loop before
        for (int j = 1; j < lags; j = j + 1) {
            temp = dP.block(i * lags + (j - 1), 0, 1, n * (ma - 1));
            temp1 = alpha * beta_t * cP.block(i * lags + (j - 1), 0, 1, n).transpose()
                    + gamma * dP.block(i * lags + (j - 1), 0, 1, n * ma).transpose();

            dP.block(i * lags + j, 0, 1, n) = temp1.transpose();
            dP.block(i * lags + j, n, 1, n * (ma - 1)) = temp;
            cP.block(i * lags + j, 0, 1, n) = cP.block(i * lags + (j - 1), 0, 1, n)
                                              + dP.block(i * lags + j, 0, 1, n);

        }
    }

    int counter = 0;
    for (int k = 0; k < lags; k = k + 1) {
        for (int l = 0; l < n; l = l + 1) {
            ccP.block(counter, 0, 1, n) = cP.block(l * lags + k, 0, 1, n);
            counter = counter + 1;
        }
    }

    for (int m = 0; m < lags; m = m + 1) {
        ccP_t.block(m * n, 0, n, n) = ccP.block(m * n, 0, n, n).transpose();
    }

    long_run_m = ccP_t.block((lags * n) - 1, 0, 1, n);

    // Calculate changes in last rows
    for (int i = 0; i < n; i = i + 1) {
        dccP_t.block(i, 0, 1, n) =
                ccP_t.block((lags * n) - 1 - (i + 1), 0, 1, n) - ccP_t.block((lags * n) - 1, 0, 1, n);
    }

    // Convergence tolerance set to 1% to check if long run convergence is reached
    for (int i = 0; i < n; i = i + 1) {
        tolerance = abs(long_run_m(0, i) * 0.01);
        if (dccP_t.block(0, i, n, 1).cwiseAbs().maxCoeff() > tolerance) {
            long_run_m(0, i) = NAN;
        }
    }

    return std::make_tuple(long_run_m, ccP_t.transpose());
}


std::tuple<MatrixXd, MatrixXd> VECM::calc_information_share(MatrixXd &long_run, MatrixXd &covariance) {
    int n = covariance.cols();

    MatrixXd S_lower = MatrixXd::Zero(1, n); // lower bound information share
    MatrixXd S_upper = MatrixXd::Zero(1, n); // upper bound information share
    MatrixXd F, share;
    MatrixXd cov_temp = covariance;
    MatrixXd long_run_temp = long_run;

    MatrixXd IS_PERM = MatrixXd::Zero(n, n);

    // Checking if long run is nan, if yes return nans for s_lower and s_upper
    if (isnan(long_run.sum())) {
        for (int i = 0; i < n; i = i + 1) {
            S_lower(0, i) = NAN;
            S_upper(0, i) = NAN;
        }

        return std::make_tuple(S_lower, S_upper);
    }

    if (fabs(long_run.sum()) < 1.0e-10) {
        std::make_tuple(MatrixXd::Zero(1, n), MatrixXd::Zero(1, n));
    }

    // Construct permutation matrix
    MatrixXd perm = MatrixXd::Zero(n, n);
    perm.block(0, 1, n - 1, n - 1) = MatrixXd::Identity(n - 1, n - 1);
    perm(n - 1, 0) = 1;

    MatrixXd total_variance = long_run * covariance * long_run.transpose();

    for (int i = 0; i < n; i = i + 1) {
        cov_temp = perm * cov_temp * perm.transpose();
        long_run_temp = long_run_temp * perm.transpose();

        // Compute Cholesky-transformation
        F = cov_temp.llt().matrixL();
        share = (long_run_temp * F).array() * (long_run_temp * F).array();

        IS_PERM.block(i, 0, 1, n) = (share * perm.pow(i + 1)) / total_variance(0, 0);
    }

    for (int i = 0; i < n; i = i + 1) {
        S_lower(0, i) = IS_PERM.block(0, i, n, 1).minCoeff();
        S_upper(0, i) = IS_PERM.block(0, i, n, 1).maxCoeff();
    }

    return std::make_tuple(S_lower, S_upper);
}


std::tuple<MatrixXd, MatrixXd> VECM::calc_edges(MatrixXd &dY, MatrixXd &dZ, MatrixXd &theta, MatrixXd &covariance,
                                                int lag_num_IRF) {

    int n = dY.rows();
    int lags = lag_num_IRF;
    MatrixXd theta_t = theta.transpose();
    MatrixXd omega = covariance;

    MatrixXd cov = MatrixXd::Zero(lags * n, n);
    MatrixXd cov_inv = MatrixXd::Zero(lags * n, n);
    MatrixXd diag = MatrixXd::Zero(lags * n, n);
    MatrixXd diag_sqrt = MatrixXd::Zero(lags * n, n);
    MatrixXd diag_sqrt_inv = MatrixXd::Zero(lags * n, n);
    MatrixXd k_matrix = MatrixXd::Identity(n, n);
    MatrixXd partial_corr_at_tau = MatrixXd::Zero(lags * n, n);
    MatrixXd lhs;
    MatrixXd temp_numerator;
    MatrixXd temp_denominator;

    for (int i = 0; i < k_matrix.rows(); i = i + 1) {
        for (int j = 0; j < k_matrix.cols(); j = j + 1) {
            if (k_matrix(i, j) == 0) {
                k_matrix(i, j) = -1;
            }
        }
    }

    // cov
    for (int i = 0; i < lags; i = i + 1) {
        cov.block(i * n, 0, n, n) =
                theta_t.block(i * n, 0, n, n) * omega * theta_t.block(i * n, 0, n, n).transpose();
        cov_inv.block(i * n, 0, n, n) = cov.block(i * n, 0, n, n).inverse();

        // Check if the covariance matrix is invertible, else return zeros matrix
        if (isinf(cov_inv(i * n, 0))) {
            cov_inv.block(i * n, 0, n, n) = MatrixXd::Zero(n, n);
        }
        diag.block(i * n, 0, n, n) = cov_inv.block(i * n, 0, n, n).diagonal().asDiagonal();

        // Check if the diagonal matrix is invertible, else return zeros matrix
        if (isinf(diag(i * n, 0))) {
            diag_sqrt_inv.block(i * n, 0, n, n) = MatrixXd::Zero(n, n);
        } else {
            diag_sqrt.block(i * n, 0, n, n) = diag.block(i * n, 0, n, n).sqrt();
            diag_sqrt_inv.block(i * n, 0, n, n) = diag_sqrt.block(i * n, 0, n, n).inverse();
        }
        lhs = diag_sqrt_inv.block(i * n, 0, n, n) * cov_inv.block(i * n, 0, n, n) *
              diag_sqrt_inv.block(i * n, 0, n, n).transpose();
        partial_corr_at_tau.block(i * n, 0, n, n) = lhs.cwiseProduct(k_matrix);
    }

    MatrixXd edges = MatrixXd::Zero(n, n);

    return std::make_tuple(edges, partial_corr_at_tau);
}


// Extending vectors
MatrixXd VECM::extend_vector(MatrixXd &vector, Matrix<bool, Dynamic, Dynamic> &included) {

    long n = included.cols();
    MatrixXd extended = MatrixXd::Constant(1, n, NAN);

    for (int i = 0, j = 0; i < n; i = i + 1) {
        if (included(i)) {
            extended(i) = vector(j);
            j = j + 1;
        }
    }
    return extended;
}

MatrixXd VECM::extend_matrix_INA(MatrixXd &mat, Matrix<bool, Dynamic, Dynamic> &included) {

    long n = included.cols();
    MatrixXd extended = MatrixXd::Constant(n, n, NAN);
    for (int i = 0, i_orig = 0; i < n; i = i + 1) {
        if (included(i)) {
            for (int j = 0, j_orig = 0; j < n; j = j + 1) {
                if (included(j)) {
                    extended(i, j) = mat(i_orig, j_orig);
                    j_orig = j_orig + 1;
                }
            }
            i_orig = i_orig + 1;
        }
    }
    return extended;
}


MatrixXd VECM::extend_matrix(MatrixXd &mat, Matrix<bool, Dynamic, Dynamic> &included) {
    long n = included.cols();
    MatrixXd extended = MatrixXd::Zero(n, n);
    for (int i = 0, i_orig = 0; i < n; i = i + 1) {
        if (included(i)) {
            for (int j = 0, j_orig = 0; j < n; j = j + 1) {
                if (included(j)) {
                    extended(i, j) = mat(i_orig, j_orig);
                    j_orig = j_orig + 1;
                }
            }
            i_orig = i_orig + 1;
        }
    }
    return extended;
}

MatrixXd VECM::extend_matrix_lag(MatrixXd &mat, Matrix<bool, Dynamic, Dynamic> &included, int lag_num_IRF, bool PI) {
    long n = included.cols();
    long lags = lag_num_IRF;
    long dim = 0;

    // Adding modularity for Price ineff and edges since their dimension are different
    if (PI) {
        dim = lags;
    } else {
        dim = lags * n;
    }
    MatrixXd extended = MatrixXd::Constant(dim, n, NAN);

    for (int i = 0, i_orig = 0; i < dim; i = i + 1) {
        // If not Price Ineffiency matrix
        if (PI == 0) {
            if (included(i % n)) {
                for (int j = 0, j_orig = 0; j < n; j = j + 1) {
                    if (included(j)) {
                        extended(i, j) = mat(i_orig, j_orig);
                        j_orig = j_orig + 1;
                    }
                }
                i_orig = i_orig + 1;
            }
            // If Price Ineffiency matrix
        } else {
            for (int j = 0, j_orig = 0; j < n; j = j + 1) {
                if (included(j)) {
                    extended(i, j) = mat(i_orig, j_orig);
                    j_orig = j_orig + 1;
                }
            }
            i_orig = i_orig + 1;
        }
    }
    return extended;
}


void VECM::print_all() {
    std::cout << "----- RESULTS (ONLY INCLUDED) -----\n";
    this->print_matrix(this->included, "Included", 1);
    this->print_matrix(this->dY, "dY", 1);
    this->print_matrix(this->Y_1, "Y_1", 1);
    this->print_matrix(this->dX, "dX", 1);
    this->print_matrix(this->long_run, "Long-run", 1);
    this->print_matrix(this->component_share, "Component share", 1);
    this->print_matrix(this->S_lower, "S_lower", 1);
    this->print_matrix(this->S_upper, "S_upper", 1);
    this->print_matrix(this->edges, "Edges Percolation", 1);
    this->print_matrix(this->edges_corr, "Edges Revelation", 1);
    this->print_matrix(this->covariance, "Covariance", 1);
    this->print_matrix(this->alpha, "Alpha", 1);
    this->print_matrix(this->beta, "Beta", 1);
    this->print_matrix(this->gamma, "Gamma", 1);
    this->print_matrix(this->theta, "Theta", 1);
    this->print_matrix(this->beta_ineff, "Beta inefficiency", 1);
    this->print_matrix(this->residuals, "Included", 1);

}


int VECM::get_num_included() {
    return this->num_included;
}

int VECM::get_dZ_num() {
    if (this->dZ.cols() > this->dZ.rows()) {
        return this->dZ.cols();
    } else {
        return this->dZ.rows();
    }

}

double *VECM::get_S_lower_ptr() {
    return this->S_lower_full.data();
}

double *VECM::get_S_upper_ptr() {
    return this->S_upper_full.data();
}

double *VECM::get_long_run_ptr() {
    return this->long_run_full.data();
}

double *VECM::get_component_share_ptr() {
    return this->component_share_full.data();
}

double *VECM::get_edges_ptr() {
    return this->edges_full.data();
}

double *VECM::get_edges_corr_ptr() {
    return this->edges_corr_full.data();
}

double *VECM::get_TE_ptr() {
    return this->TE_full.data();
}

double *VECM::get_TE_pval_ptr() {
    return this->TE_pval_full.data();
}

double *VECM::get_CTE_ptr() {
    return this->CTE_full.data();
}

double *VECM::get_CTE_pval_ptr() {
    return this->CTE_pval_full.data();
}

double *VECM::get_covariance_ptr() {
    return this->covariance_full.data();
}

double *VECM::get_data_ptr() {
    return this->data.data();
}

int *VECM::get_included_ptr() {
    return this->included_int.data();
}

MatrixXd VECM::get_S_lower() {
    return this->S_lower_full;
}

MatrixXd VECM::get_S_upper() {
    return this->S_upper_full;
}

MatrixXd VECM::get_long_run() {
    return this->long_run_full;
}

MatrixXd VECM::get_component_share() {
    return this->component_share_full;
}

MatrixXd VECM::get_edges() {
    return this->edges_full;
}

MatrixXd VECM::get_edges_corr() {
    return this->edges_corr_full;
}

MatrixXd VECM::get_TE() {
    return this->TE_full;
}

MatrixXd VECM::get_TE_pval() {
    return this->TE_pval_full;
}

MatrixXd VECM::get_CTE() {
    return this->CTE_full;
}

MatrixXd VECM::get_CTE_pval() {
    return this->CTE_pval_full;
}

MatrixXd VECM::get_covariance() {
    return this->covariance_full;
}

MatrixXi VECM::get_included() {
    return this->included_int;
}

double *VECM::get_alpha_ptr() {
    return this->alpha.data();
}

double *VECM::get_pi_ptr() {
    return this->pi.data();
}

double *VECM::get_beta_ptr() {
    return this->beta.data();
}

double *VECM::get_gamma_ptr() {
    return this->gamma.data();
}

double *VECM::get_theta_ptr() {
    return this->theta.data();
}

double *VECM::get_dZ_ptr() {
    return this->dZ.data();
}

double *VECM::get_residuals_ptr() {
    return this->residuals.data();
}

MatrixXd VECM::get_residuals() {
    return this->residuals;
}

MatrixXd VECM::get_alpha() {
    return this->alpha;
}

MatrixXd VECM::get_pi() {
    return this->pi;
}

MatrixXd VECM::get_beta() {
    return this->beta;
}

MatrixXd VECM::get_gamma() {
    return this->gamma;
}

MatrixXd VECM::get_theta() {
    return this->theta;
}

double *VECM::get_beta_ineff_ptr() {
    return this->beta_ineff_full.data();
}

MatrixXd VECM::get_beta_ineff() {
    return this->beta_ineff_full;
}

MatrixXd VECM::get_dZ() {
    return this->dZ;
}


