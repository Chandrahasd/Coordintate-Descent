/*
 * LogisticLoss.h
 *
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#ifndef LOGISTICLOSS_H_
#define LOGISTICLOSS_H_

#include "Function.h"

class LogisticLoss: public Function {
public:
	Eigen::MatrixXd A,Y;	//A: Data matrix, Y: Labels
	long int m,n,omega;		//m: number of examples, n:dimension, omega: degree of partial separability
	double regularizer;		//dummy variable
	LogisticLoss();
	LogisticLoss(Eigen::MatrixXd, Eigen::MatrixXd);
	LogisticLoss(Eigen::MatrixXd, Eigen::MatrixXd, long int=-1);
	double eval(Eigen::VectorXd, Eigen::VectorXd);				//Used for calculating function value at given point
	Eigen::VectorXd grad(Eigen::VectorXd, Eigen::VectorXd);		//Used for calculating function gradient at given point
	double grad_i(Eigen::VectorXd, Eigen::VectorXd, long int);	//Used for calculating function gradient at given point for given coordinate
	double grad_b(Eigen::VectorXd, Eigen::VectorXd);			//Used for calculating function gradient wrt bias term, ignored in current implementation
	virtual ~LogisticLoss();
};

#endif /* LOGISTICLOSS_H_ */
