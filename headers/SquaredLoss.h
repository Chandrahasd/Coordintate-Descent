/*
 * SquaredLoss.h
 * This implements Squared loss and its gradients
 *  Created on: 27-Oct-2015
 *      Author: chandrahas
 */

#ifndef SQUAREDLOSS_H_
#define SQUAREDLOSS_H_

#include "Function.h"

class SquaredLoss: public Function {
public:
	Eigen::MatrixXd A, Y; // A: data matrix with Bias, Y: Label Matrix
	long int m,n,omega;  // m : number of examples, n:dimension, omega: degree of partial separability
	double regularizer; // dummy variable
	SquaredLoss();
	SquaredLoss(Eigen::MatrixXd, Eigen::MatrixXd, long int=-1);
	double eval(Eigen::VectorXd,  Eigen::VectorXd); 		//Used for calculating function value at given point
	Eigen::VectorXd grad(Eigen::VectorXd,  Eigen::VectorXd);//Used for calculating function gradient at given point
	double grad_i(Eigen::VectorXd, Eigen::VectorXd, int);   //Used for calculating function gradient at given point for given coordinate
	double grad_b(Eigen::VectorXd, Eigen::VectorXd);        //Gradient wrt bias, ignored in current implementation
	virtual ~SquaredLoss();
};

#endif /* SQUAREDLOSS_H_ */
