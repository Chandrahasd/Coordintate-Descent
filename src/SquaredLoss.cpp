/*
 * SquaredLoss.cpp
 *
 *  Created on: 27-Oct-2015
 *      Author: chandrahas
 */

#include <iostream>
#include "Eigen/Dense"
#include "SquaredLoss.h"

SquaredLoss::SquaredLoss() {
	this->A = Eigen::MatrixXd::Random(100,90);
	this->Y = Eigen::MatrixXd::Random(100,1);
}

SquaredLoss::~SquaredLoss() {
}

SquaredLoss::SquaredLoss(Eigen::MatrixXd x, Eigen::MatrixXd y, long int w){
	this->A = x;
	this->Y = y;
	this->m = x.innerSize();
	this->n = x.outerSize();
	this->regularizer = 0.0;
	if(w <= 0){
		this->omega = (this->A).outerSize();
	}
	else{
		this->omega = w;
	}
}

double SquaredLoss::eval( Eigen::VectorXd x, Eigen::VectorXd Ax){
	double t = (Ax - Y).transpose()*(Ax - Y);
	t = t/(this->m);
    return 0.5*t ;
}

Eigen::VectorXd SquaredLoss::grad( Eigen::VectorXd x, Eigen::VectorXd Ax){
	Eigen::VectorXd g = (this->A).transpose()*(Ax - Y);
	g = g/(this->m);
	return g;
}

double SquaredLoss::grad_i( Eigen::VectorXd x, Eigen::VectorXd Ax, int i){
	double g = ((this->A).col(i)).transpose() * (Ax-Y) ;
	g = g/(this->m);
	return g;
}

double SquaredLoss::grad_b( Eigen::VectorXd x, Eigen::VectorXd Ax){
	VectorXd g = Ax-Y ;
	g = g/(this->m);
	return g.sum();
}
