#include <set>
#include "DailyVECM.h"
#include <thread>
#include <iostream>
#include "VECM.h"
#include <stdio.h>

DailyVECM::DailyVECM() = default;


DailyVECM::DailyVECM(int *time, int *market_id, double *ask, double *bid, int n) {

    this->time = time;
    this->market_id = market_id;
    this->ask = ask;
    this->bid = bid;
    this->n = n;

    // Count the number of venues
    std::set<int> market_id_set;
    for (int i = 0; i < n; i = i + 1) {
        market_id_set.insert(market_id[i]);
    }
    this->num_markets = market_id_set.size();
}

void DailyVECM::init(int fs_ms, int window_size_ms, int step_size_ms, int begin_ms, int end_ms) {
    // Initialize dataset
    this->fs_ms = (1 <= fs_ms) ? fs_ms : 25;
    this->step_size_num = (fs_ms <= step_size_ms) ? step_size_ms / fs_ms : 1;
    this->window_length_num = window_size_ms / fs_ms;

    int rows = (end_ms - begin_ms) / fs_ms + 1;
    this->mid = MatrixXd::Constant(rows, this->num_markets, -1);
    this->spread = MatrixXd::Constant(rows, this->num_markets, -1);

    // Set up the time vector
    this->time_vector = MatrixXi(1, rows);
    this->time_vector(0, 0) = begin_ms;
    for (int i = 1; i < rows; i = i + 1) {
        this->time_vector(0, i) = this->time_vector(0, i - 1) + fs_ms;
    }
    this->time_vector.array() = this->time_vector.array() + window_size_ms;

    // Collect prices in regular intervals
    int index, correction;
    for (int i = 0; i < this->n; i = i + 1) {
        if ((begin_ms <= time[i]) && (time[i] < end_ms)) {
            correction = 1 - ((fs_ms != 1) && (time[i] == begin_ms));
            index = (time[i] - begin_ms - 1) / fs_ms + correction;
            this->mid(index, market_id[i]) = (ask[i] + bid[i]) / 2;
            this->spread(index, market_id[i]) = ask[i] - bid[i];
        }
    }

    // Fill unknown values (forward)
    for (int i = 0; i < this->num_markets; i = i + 1) {
        for (int j = 1; j < rows; j = j + 1) {
            if (this->mid(j, i) == -1) {
                this->mid(j, i) = this->mid(j - 1, i);
                this->spread(j, i) = this->spread(j - 1, i);
            }
        }
    }

    // Fill unknown values (backward) -- barely ever gets executed since all values are already filled with forward
    for (int i = 0; i < this->num_markets; i = i + 1) {
        for (int j = rows - 1; j >= 0; j = j - 1) {
            if (this->mid(j, i) == -1) {
                this->mid(j, i) = this->mid(j + 1, i);
                this->spread(j, i) = this->spread(j + 1, i);
            }
        }
    }

}


void DailyVECM::run(int num_threads,
                    int lag_num_VECM,
                    int lag_num_IRF,
                    int min_obs,
                    int matrix_transform,
                    int TE_hist,
                    int TE_tau,
                    int TE_permutation,
                    int TE_auto) {

    // Create container for the starting indexes
    int num_results = 0;
    for (int i = 0; i + this->window_length_num < this->mid.rows(); i = i + this->step_size_num) {
        num_results = num_results + 1;
    }

    // We need to store from which rows will each worker work on a window
    MatrixXi start_index = MatrixXi::Zero(1, num_results);
    int temp = 0;
    for (int i = 0; i + this->window_length_num < this->mid.rows(); i = i + this->step_size_num) {
        start_index(temp) = i;
        temp = temp + 1;
    }

    // Initialize containers that will store results.
    // These are passed as references to the threads which will fill them
    this->res_time = MatrixXd::Zero(1, num_results);
    this->res_spread = MatrixXd::Zero(num_results, this->num_markets);
    this->res_variance = MatrixXd::Zero(num_results, this->num_markets);
    this->res_S_lower = MatrixXd::Zero(num_results, this->num_markets);
    this->res_S_upper = MatrixXd::Zero(num_results, this->num_markets);
    this->res_long_run = MatrixXd::Zero(num_results, this->num_markets);
    this->res_component_share = MatrixXd::Zero(num_results, this->num_markets);
    this->res_edges = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);
    this->res_covariance = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);
    this->res_included = MatrixXi::Zero(num_results, this->num_markets);
    this->res_edges_corr = MatrixXd::Zero(num_results, this->num_markets * num_markets * lag_num_IRF);
    this->res_beta_ineff = MatrixXd::Zero(num_results, this->num_markets * lag_num_IRF);
    this->res_TE = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);
    this->res_TE_pval = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);
    this->res_CTE = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);
    this->res_CTE_pval = MatrixXd::Zero(num_results, this->num_markets * this->num_markets);

    // Limit the number of threads
    std::thread myThreads[52];
    if (num_threads > 52) {
        num_threads = 52;
    }

    // Start the threads, and pass the result containers to store the results
    for (int i = 0; i < num_threads; i = i + 1) {
        myThreads[i] = std::thread(&DailyVECM::worker,
                                   this,
                                   i,
                                   num_threads,
                                   std::ref(this->mid),
                                   std::ref(this->spread),
                                   std::ref(this->time_vector),
                                   std::ref(start_index),
                                   this->window_length_num,
                                   std::ref(this->res_time),
                                   std::ref(this->res_spread),
                                   std::ref(this->res_variance),
                                   std::ref(this->res_S_lower),
                                   std::ref(this->res_S_upper),
                                   std::ref(this->res_long_run),
                                   std::ref(this->res_component_share),
                                   std::ref(this->res_edges),
                                   std::ref(this->res_edges_corr),
                                   std::ref(this->res_covariance),
                                   std::ref(this->res_beta_ineff),
                                   std::ref(this->res_included),
                                   std::ref(this->res_TE),
                                   std::ref(this->res_TE_pval),
                                   std::ref(this->res_CTE),
                                   std::ref(this->res_CTE_pval),
                                   lag_num_VECM,
                                   lag_num_IRF,
                                   min_obs,
                                   matrix_transform,
                                   TE_hist,
                                   TE_tau,
                                   TE_permutation,
                                   TE_auto);

    }
    // Join all threads when they finish
    for (int i = 0; i < num_threads; i = i + 1) {
        myThreads[i].join();
    }

}

void DailyVECM::worker(int id,
                       int num_workers,
                       MatrixXd &mid,
                       MatrixXd &spread,
                       MatrixXi &time_vector,
                       MatrixXi &start_index,
                       int window_length_num,
                       MatrixXd &res_time,
                       MatrixXd &res_spread,
                       MatrixXd &res_variance,
                       MatrixXd &res_S_lower,
                       MatrixXd &res_S_upper,
                       MatrixXd &res_long_run,
                       MatrixXd &res_component_share,
                       MatrixXd &res_edges,
                       MatrixXd &res_edges_corr,
                       MatrixXd &res_covariance,
                       MatrixXd &res_beta_ineff,
                       MatrixXi &res_included,
                       MatrixXd &res_TE,
                       MatrixXd &res_TE_pval,
                       MatrixXd &res_CTE,
                       MatrixXd &res_CTE_pval,
                       int lag_num_VECM,
                       int lag_num_IRF,
                       int min_obs,
                       int matrix_transform,
                       int TE_hist,
                       int TE_tau,
                       int TE_permutation,
                       int TE_auto) {


    for (int i = id; i < start_index.cols(); i = i + num_workers) {

        // Get proper slices form the original table
        MatrixXd mid_slice = mid.middleRows(start_index(i), window_length_num);
        MatrixXd spread_slice = spread.middleRows(start_index(i), window_length_num);

        // Compute the average spread
        MatrixXd avg_spread = spread_slice.colwise().mean();

        // Compute variance
        MatrixXd mid_diff = mid_slice.topRows(mid_slice.rows() - 1) - mid_slice.bottomRows(mid_slice.rows() - 1);
        MatrixXd centered = mid_diff.rowwise() - mid_diff.colwise().mean();
        MatrixXd variance = (centered.array() * centered.array()).colwise().mean();

        // Initialize the VEC model
        VECM vecmodel(mid_slice.data(), mid_slice.rows(), mid_slice.cols(), lag_num_VECM,
                      lag_num_IRF, min_obs, TE_hist, TE_tau, TE_permutation, TE_auto);

        // Estimate the parameters
        vecmodel.run_models(matrix_transform);

        // Store results in the containers
        res_time(i) = (double) time_vector(start_index(i));
        res_spread.row(i) = avg_spread;
        res_variance.row(i) = variance;
        res_S_lower.row(i) = vecmodel.get_S_lower();
        res_S_upper.row(i) = vecmodel.get_S_upper();
        res_long_run.row(i) = vecmodel.get_long_run();
        res_component_share.row(i) = vecmodel.get_component_share();
        res_included.row(i) = vecmodel.get_included();


        // Flatten matrices
        MatrixXd edges = vecmodel.get_edges();
        MatrixXd covariance = vecmodel.get_covariance();
        MatrixXd TE_preflat = vecmodel.get_TE();
        MatrixXd TE_pval_preflat = vecmodel.get_TE_pval();
        MatrixXd CTE_preflat = vecmodel.get_CTE();
        MatrixXd CTE_pval_preflat = vecmodel.get_CTE_pval();

        int k0 = 0;
        for (int k1 = 0; k1 < edges.rows(); k1 = k1 + 1) {
            for (int k2 = 0; k2 < edges.rows(); k2 = k2 + 1) {
                res_edges(i, k0) = edges(k1, k2);
                res_covariance(i, k0) = covariance(k1, k2);
                res_TE(i, k0) = TE_preflat(k1, k2);
                res_TE_pval(i, k0) = TE_pval_preflat(k1, k2);
                res_CTE(i, k0) = CTE_preflat(k1, k2);
                res_CTE_pval(i, k0) = CTE_pval_preflat(k1, k2);

                k0 = k0 + 1;
            }
        }

        // Flattening matrices for EdgesCorr and BetaInefficiency - different dimensions
        MatrixXd edges_corr = vecmodel.get_edges_corr();  // DIM [ n*lags, n]

        int c0 = 0;
        for (int c1 = 0; c1 < edges_corr.rows(); c1 = c1 + 1) {
            for (int c2 = 0; c2 < edges_corr.cols(); c2 = c2 + 1) {
                res_edges_corr(i, c0) = edges_corr(c1, c2);
                c0 = c0 + 1;
            }
        }

        MatrixXd beta_ineff = vecmodel.get_beta_ineff(); // DIM [ lags, n]
        int d0 = 0;
        for (int d1 = 0; d1 < beta_ineff.rows(); d1 = d1 + 1) {
            for (int d2 = 0; d2 < beta_ineff.cols(); d2 = d2 + 1) {
                res_beta_ineff(i, d0) = beta_ineff(d1, d2);
                d0 = d0 + 1;
            }
        }
    }
}

double *DailyVECM::get_time_ptr() {
    return this->res_time.data();
}

int *DailyVECM::get_included_ptr() {
    return this->res_included.data();
}

double *DailyVECM::get_spread_ptr() {
    return this->res_spread.data();
}

double *DailyVECM::get_variance_ptr() {
    return this->res_variance.data();
}

double *DailyVECM::get_S_lower_ptr() {
    return this->res_S_lower.data();
}

double *DailyVECM::get_S_upper_ptr() {
    return this->res_S_upper.data();
}

double *DailyVECM::get_long_run_ptr() {
    return this->res_long_run.data();
}

double *DailyVECM::get_component_share_ptr() {
    return this->res_component_share.data();
}

double *DailyVECM::get_edges_ptr() {
    return this->res_edges.data();
}

double *DailyVECM::get_edges_corr_ptr() {
    return this->res_edges_corr.data();
}

double *DailyVECM::get_TE_ptr() {
    return this->res_TE.data();
}

double *DailyVECM::get_TE_pval_ptr() {
    return this->res_TE_pval.data();
}

double *DailyVECM::get_CTE_ptr() {
    return this->res_CTE.data();
}

double *DailyVECM::get_CTE_pval_ptr() {
    return this->res_CTE_pval.data();
}

double *DailyVECM::get_covariance_ptr() {
    return this->res_covariance.data();
}

double *DailyVECM::get_beta_ineff_ptr() {
    return this->res_beta_ineff.data();
}

int DailyVECM::get_num_markets() {
    return this->res_included.cols();
}

int DailyVECM::get_num_results() {
    return this->res_included.rows();
}

