//
// Created by chandrahas on 17/9/15.
//

#include "SteepestGradientDescent.h"
#include "Function.h"
#include "QuadraticForm.h"
#include<iostream>
#include<Eigen/Dense>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

SteepestGradientDescent::SteepestGradientDescent() {
    learningRate = 0.01;
    threshold = 1;
    QuadraticForm *f = new QuadraticForm();
}
SteepestGradientDescent::SteepestGradientDescent(double alpha, double epsilon, QuadraticForm *obj) {
    learningRate = alpha;
    threshold = epsilon;
    f = obj;
}
double SteepestGradientDescent::optimize() {
    int T = 1000;
    int t = 0;
    int n = f->getDimension();
    VectorXd x = VectorXd::Random(n);
    VectorXd g = f->grad(x);
    while(t<T && g.norm() > threshold){
        x = x - learningRate*g;
        t++;
        g = f->grad(x);
        std::cout << "OBJ: " << f->eval(x) << std::endl ;
    }
    std::cout << "Min point is    : " << x << std::endl;
    std::cout << "Gradient is     : " << g << std::endl;
    std::cout << "Function val is : " << f->eval(x) << std::endl ;
    return f->eval(x);
}
