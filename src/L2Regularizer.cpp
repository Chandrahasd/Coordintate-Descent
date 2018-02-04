/*
 * L2Regularizer.cpp
 * Appends L2 regularizer to base loss function
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#include <iostream>
#include "L2Regularizer.h"
#include "Eigen/Dense"

template<class FunctionType>
L2Regularizer<FunctionType>::L2Regularizer(): FunctionType(), regularizer(1.0) {
	//unused constructor
}

template<class FunctionType>
L2Regularizer<FunctionType>::L2Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y) : FunctionType(x, y), regularizer(1.0) {
}

template<class FunctionType>
L2Regularizer<FunctionType>::L2Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y, double lambda) : FunctionType(x, y), regularizer(lambda) {
}

template<class FunctionType>
L2Regularizer<FunctionType>::L2Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y, long int w, double lambda) : FunctionType(x, y, w), regularizer(lambda) {
}

template<class FunctionType>
double L2Regularizer<FunctionType>::eval(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	double baseval = FunctionType::eval(x, Ax);
	double regval = x.norm();
	//exclude the bias term from regularizer
	regval = regval*regval - x(0)*x(0);
	return baseval + (this->regularizer)*0.5*regval ;
}

template<class FunctionType>
Eigen::VectorXd L2Regularizer<FunctionType>::grad(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	Eigen::VectorXd baseval = FunctionType::grad(x, Ax);
	Eigen::VectorXd regval = (this->regularizer)*x;
	//exclude bias term from regularizer
	regval(0) = 0;
	return baseval + regval;
}

template<class FunctionType>
double L2Regularizer<FunctionType>::grad_i(Eigen::VectorXd x, Eigen::VectorXd Ax, int i) {
	double baseval = FunctionType::grad_i(x, Ax, i);
	//exclude bias term from regularizer
	return baseval + (i!=0)*(this->regularizer)*x(i);
}

template<class FunctionType>
double L2Regularizer<FunctionType>::grad_b(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	double baseval = FunctionType::grad_b(x, Ax);
	return baseval;
}

template<class FunctionType>
L2Regularizer<FunctionType>::~L2Regularizer() {
}
