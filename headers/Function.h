//
// Created by chandrahas on 17/9/15.
//

#ifndef OPTIMIZATION_FUNCTION_H
#define OPTIMIZATION_FUNCTION_H

#include "Eigen/Dense"

using Eigen::VectorXd;
using Eigen::MatrixXd;

class Function {
private:
    double val;
public:
    double eval(VectorXd x);
    VectorXd grad(VectorXd x);
    MatrixXd hessian(VectorXd x);
    int getDimension();
};


#endif //OPTIMIZATION_FUNCTION_H
