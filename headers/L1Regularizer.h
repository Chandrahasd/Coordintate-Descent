/*
 * L1Regularizer.h
 *
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#ifndef L1REGULARIZER_H_
#define L1REGULARIZER_H_

#include "Function.h"

template<class FunctionType>
class L1Regularizer: public FunctionType {
public:
	double regularizer;
	L1Regularizer();
	virtual ~L1Regularizer();
	L1Regularizer(Eigen::MatrixXd, Eigen::MatrixXd);
	L1Regularizer(Eigen::MatrixXd, Eigen::MatrixXd, double);
	L1Regularizer(Eigen::MatrixXd, Eigen::MatrixXd, long int, double);
	double eval(Eigen::VectorXd, Eigen::VectorXd);			//Used to calculate function value at given point
	Eigen::VectorXd grad(Eigen::VectorXd, Eigen::VectorXd); //Used to calculate function gradient at given point
	double grad_i(Eigen::VectorXd, Eigen::VectorXd, int);	//Used to calculate function gradient at given point for given coordinate
	double grad_b(Eigen::VectorXd, Eigen::VectorXd);		//Gradient wrt bias term, ignored in current implementation
};

#endif /* L1REGULARIZER_H_ */
