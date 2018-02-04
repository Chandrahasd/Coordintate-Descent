/*
 * L2Regularizer.h
 *
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#ifndef L2REGULARIZER_H_
#define L2REGULARIZER_H_

#include "Function.h"

template<class FunctionType>
class L2Regularizer: public FunctionType {
//class L2Regularizer{
public:
	//FunctionType loss;
	double regularizer;
	L2Regularizer();
	L2Regularizer(Eigen::MatrixXd, Eigen::MatrixXd);
	L2Regularizer(Eigen::MatrixXd, Eigen::MatrixXd, double);
	L2Regularizer(Eigen::MatrixXd, Eigen::MatrixXd, long int, double);
	//L2Regularizer(FunctionType);
	//L2Regularizer(FunctionType, double);
	double eval(Eigen::VectorXd, Eigen::VectorXd);			//Used to calculate function value at given point
	Eigen::VectorXd grad(Eigen::VectorXd, Eigen::VectorXd); //Used to calculate function gradient at given point
	double grad_i(Eigen::VectorXd, Eigen::VectorXd, int);   //Used to calculate function gradient at given point for given coordinate
	double grad_b(Eigen::VectorXd, Eigen::VectorXd); 		//Gradient wrt bias term, ignored in current implementation
	virtual ~L2Regularizer();
};

#endif /* L2REGULARIZER_H_ */
