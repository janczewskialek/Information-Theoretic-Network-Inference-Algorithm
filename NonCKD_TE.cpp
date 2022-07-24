//
// Created by Aleksander on 2/4/22.
//

#include "NonCKD_TE.h"
#include <time.h>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>

#include "circ_shift.h"
#include <chrono>
#include "ckdtree/src/ckdtree_decl.h"
#include <algorithm>    
#include <fstream>


// Function used for calculating distance at k=3 and NN of specific radius
double chebyshev_distance_double(const double *u, const double *v, const int n)
{
    double maxv = 0.0;

    for (int i = 0; i < n; ++i) {
        const double d = fabs(u[i] - v[i]);
        if (d > maxv) {
            maxv = d;
        }
    }
    return maxv;
}

int cdist_cheb(double *XA, double *XB, double *dm, const int num_rowsA, const int num_rowsB,
               const int num_cols)
{

    for (int i = 0; i < num_cols; ++i) {
        double *u = &XA[num_rowsA * i];

        for (int j = 0; j < num_cols; ++j, ++dm) {
            double *v = &XB[num_rowsB * j];
            *dm = chebyshev_distance_double(u, v, num_rowsB);
        }
    }
    return 0;
}



double chebyshev_distance_doubleNN(const double *u, const double *v, const int n, const double *md)
{
    double maxv = 0.0;

    for (int i = 0; i < n; ++i) {
        const double d = fabs(u[i] - v[i]);
        if (d > maxv) {
            maxv = d;
        }
        if (maxv >= *md){break;}; // if in any dimension is greater than max dist then its not a neighbour --> break
    }
    return maxv;
}

int cdist_chebNN(double *XA, double *XB, double *dm, double*max_dist, const int num_rowsA, const int num_rowsB,
                 const int num_cols)
{

    for (int i = 0; i < num_cols; ++i, ++dm) {
        double NN_counter = 0.0;
        double *u = &XA[num_rowsA * i];
        double *md = &max_dist[i];

        for (int j = 0; j < num_cols; ++j) {
            double *v = &XB[num_rowsB * j];
            if (chebyshev_distance_doubleNN(u, v, num_rowsB, md) < *md){
                NN_counter +=1;
            }
        }
        *dm = NN_counter;
    }
    return 0;
}

void sort_Matrix (Eigen::MatrixXd& MATRIX){
    for (int i=0; i<MATRIX.cols(); i++)
    {
        std::sort(MATRIX.block(0,i,MATRIX.rows(),1).array().data(),
                  MATRIX.block(0,i,MATRIX.rows(),1).array().data()+
                  MATRIX.block(0,i,MATRIX.rows(),1).array().size());

    }
} // pretty fast



Eigen::MatrixXd count_NNs_with_radius_search(Eigen::MatrixXd& Y_shift, Eigen::MatrixXd& Y, Eigen::MatrixXd& dvec){
    using namespace Eigen;
    assert(Y_shift.rows() == Y.rows());
    assert(Y_shift.cols() == Y.cols());

    // Y1Y STACK
    int rowsY1Y = Y_shift.rows() + Y.rows();
    MatrixXd Y1Y_stack = MatrixXd::Zero(rowsY1Y, Y_shift.cols());
    Y1Y_stack.block(0,0,1,Y_shift.cols()) = Y_shift;
    Y1Y_stack.block(1,0,1,Y.cols()) = Y;

    MatrixXd NNcount_Y1Y = MatrixXd::Zero(dvec.rows(), dvec.cols());
    cdist_chebNN(Y1Y_stack.data(), Y1Y_stack.data(), NNcount_Y1Y.data(), dvec.data(), Y1Y_stack.rows(),Y1Y_stack.rows(), Y1Y_stack.cols());



    return NNcount_Y1Y;
}



double TE_alldist_from_X_to_Y(Eigen::MatrixXd& X_raw, Eigen::MatrixXd& Y_raw, const int k=3, const int X_hist=1, const int Y_hist=1, const int truncation=0){

    using namespace Eigen;
    double base = exp(1);

    MatrixXd X_raw_shift = circShift(X_raw, 0, -1);
    MatrixXd Y_raw_shift = circShift(Y_raw, 0, -1);

    // getting rid of last values because of the shift applied
    MatrixXd X = X_raw.block(0,0,1,X_raw.cols()-truncation);
    MatrixXd Y = Y_raw.block(0,0,1,Y_raw.cols()-truncation);
    MatrixXd X_shift = X_raw_shift.block(0,0,1,X_raw_shift.cols()-truncation);
    MatrixXd Y_shift = Y_raw_shift.block(0,0,1,Y_raw_shift.cols()-truncation);

    // assertions
    assert(X.rows() == Y.rows());
    assert(X.cols() == Y.cols());
    assert(Y_shift.rows() == Y.rows());
    assert(Y_shift.cols() == Y.cols());


    int rows = Y_shift.rows() + X.rows() + Y.rows();
    MatrixXd Y1XY_stack = MatrixXd::Zero(rows, Y_shift.cols());
    Y1XY_stack.block(0,0,1,Y_shift.cols()) = Y_shift;
    Y1XY_stack.block(1,0,1,X.cols()) = X;
    Y1XY_stack.block(2,0,1,Y.cols()) = Y;

    // PART 2
    // NEED A FILLER ZEROS FOR K>L
    MatrixXd dvec_full = MatrixXd::Zero(Y1XY_stack.cols(),Y1XY_stack.cols());

    MatrixXd closest_NNS_block = MatrixXd::Zero(k+1, Y1XY_stack.cols());


    cdist_cheb(Y1XY_stack.data(), Y1XY_stack.data(), dvec_full.data(), Y1XY_stack.rows(), Y1XY_stack.rows(), Y1XY_stack.cols());



    sort_Matrix(dvec_full);


    MatrixXd dvec = dvec_full.block(k, 0, 1,dvec_full.cols());
    MatrixXd NNcount_Y1Y = count_NNs_with_radius_search(Y_shift, Y, dvec);
    MatrixXd NNcount_XY  = count_NNs_with_radius_search(X, Y, dvec);

    MatrixXd Y_filler   = MatrixXd::Zero(Y.rows(), Y.cols());
    MatrixXd NNcount_Y  = count_NNs_with_radius_search(Y, Y_filler, dvec);


    MatrixXd k_matrix = MatrixXd::Zero(1,1);
    k_matrix(0,0) = k;

    double Digamma_A = digamma(NNcount_Y1Y.array()).mean();
    double Digamma_B = digamma(NNcount_XY.array()).mean();
    double Digamma_C = digamma(NNcount_Y.array()).mean();
    double Digamma_D = digamma(k_matrix.array()).mean();




    double TE_XtoY_Kraskov = (-Digamma_A - Digamma_B + Digamma_C + Digamma_D) / log(base);

    return TE_XtoY_Kraskov;
}
