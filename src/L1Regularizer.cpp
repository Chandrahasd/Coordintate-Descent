/*
 * L1Regularizer.cpp
 * Appends L1 regularizer to base loss function
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#include "L1Regularizer.h"
#include "Eigen/Dense"

template<class FunctionType>
L1Regularizer<FunctionType>::L1Regularizer():FunctionType(), regularizer(1.0) {
}

template<class FunctionType>
L1Regularizer<FunctionType>::L1Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y):FunctionType(x, y), regularizer(1.0) {
}

template<class FunctionType>
L1Regularizer<FunctionType>::L1Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y, double lambda) : FunctionType(x, y), regularizer(lambda) {
}

template<class FunctionType>
L1Regularizer<FunctionType>::L1Regularizer(Eigen::MatrixXd x, Eigen::MatrixXd y, long int w, double lambda) : FunctionType(x, y, w), regularizer(lambda) {
}

template<class FunctionType>
double L1Regularizer<FunctionType>::eval(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	double baseval = FunctionType::eval(x, Ax);
	double regval = x.cwiseAbs().sum();
	//exclude bias term
	regval = regval - x.cwiseAbs()(0);
	return baseval + (this->regularizer)*regval ;
}

template<class FunctionType>
Eigen::VectorXd L1Regularizer<FunctionType>::grad(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	Eigen::VectorXd baseval = FunctionType::grad(x, Ax);
	Eigen::VectorXd regval = x;
	for(int i=0;i<x.innerSize();i++){
		if(regval(i) < 0)
			regval(i) = -1;
		else if(regval(i) > 0)
			regval(i) = 1;
		else
			regval(i) = 0;
	}
	regval(0) = 0;
	return baseval + x;
}

template<class FunctionType>
double L1Regularizer<FunctionType>::grad_i(Eigen::VectorXd x, Eigen::VectorXd Ax, int i) {
	double baseval = FunctionType::grad_i(x, Ax, i);
	if(x(i) > 0)
		return baseval + (i!=0)*1;
	else if(x(i) < 0)
		return baseval - (i!=0)*1;
	else
		return 0;
}

template<class FunctionType>
double L1Regularizer<FunctionType>::grad_b(Eigen::VectorXd x, Eigen::VectorXd Ax) {
	double baseval = FunctionType::grad_b(x, Ax);
	return baseval;
}

template<class FunctionType>
L1Regularizer<FunctionType>::~L1Regularizer() {
}
