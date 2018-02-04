//
// Created by chandrahas on 17/9/15.
//

#ifndef OPTIMIZATION_QUADRATICFORM_H
#define OPTIMIZATION_QUADRATICFORM_H

#include <cmath>
#include <iostream>
#include "Eigen/Dense"
#include "Function.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class QuadraticForm:public Function {
private:
    MatrixXd Q;
    VectorXd b;
    int dimension;
public:
    QuadraticForm();
    QuadraticForm(MatrixXd Q1, MatrixXd b1);
    double eval(VectorXd x);
    VectorXd grad(VectorXd x);
    MatrixXd hessian(VectorXd x);
    int getDimension();
};


#endif //OPTIMIZATION_QUADRATICFORM_H
