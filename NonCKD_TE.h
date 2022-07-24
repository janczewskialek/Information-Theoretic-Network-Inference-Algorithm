//
// Created by Aleksander on 2/4/22.
//

#ifndef TRANSFERENTROPY_NONCKD_TE_H
#define TRANSFERENTROPY_NONCKD_TE_H



double chebyshev_distance_double(const double *u, const double *v, const int n);

int cdist_cheb(double *XA, double *XB, double *dm, const int num_rowsA, const int num_rowsB,
               const int num_cols);

double chebyshev_distance_doubleNN(const double *u, const double *v, const int n, const double *md);

int cdist_chebNN(double *XA, double *XB, double *dm, double*max_dist, const int num_rowsA, const int num_rowsB,
                 const int num_cols);

void sort_Matrix (Eigen::MatrixXd& MATRIX);

Eigen::MatrixXd count_NNs_with_radius_search(Eigen::MatrixXd& Y_shift, Eigen::MatrixXd& Y, Eigen::MatrixXd& dvec);

double TE_alldist_from_X_to_Y(Eigen::MatrixXd& X_raw, Eigen::MatrixXd& Y_raw, const int k=3, const int X_hist=1, const int Y_hist=1, const int truncation=0);



#endif //TRANSFERENTROPY_NONCKD_TE_H
