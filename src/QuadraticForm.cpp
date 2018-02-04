//
// Created by chandrahas on 17/9/15.
//

#include<Eigen/Dense>
#include "QuadraticForm.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;


QuadraticForm::QuadraticForm() {
    Q = MatrixXd::Random(2,2);
    b = VectorXd::Random(2);
    dimension = 2;
}
QuadraticForm::QuadraticForm(MatrixXd Q1, MatrixXd b1) {
    Q = Q1;
    b = b1;
    dimension = (int)b.size();
}
double QuadraticForm::eval(VectorXd x){
    double val = 0.5 * x.transpose()*(Q*x) - b.dot(x) ;
    return val;
}
VectorXd QuadraticForm::grad(VectorXd x){
    VectorXd val = (Q*x) - b;
    return val;
}
MatrixXd QuadraticForm::hessian(VectorXd x){
    return Q;
}

int QuadraticForm::getDimension() {
    return dimension;
}