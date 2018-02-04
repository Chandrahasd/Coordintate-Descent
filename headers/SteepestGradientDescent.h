//
// Created by chandrahas on 17/9/15.
//

#ifndef OPTIMIZATION_STEEPESTGRADIENTDESCENT_H
#define OPTIMIZATION_STEEPESTGRADIENTDESCENT_H

#include<iostream>
#include "Eigen/Dense"
#include "Function.h"
#include "QuadraticForm.h"

using namespace std;
using Eigen::VectorXd;

class SteepestGradientDescent {
private:
    double learningRate;
    double threshold;
    QuadraticForm *f;
public:
    SteepestGradientDescent();
    SteepestGradientDescent(double alpha, double epsilon, QuadraticForm *obj);
    double optimize();
};


#endif //OPTIMIZATION_STEEPESTGRADIENTDESCENT_H
