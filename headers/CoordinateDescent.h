/*
 * CoordinateDescent.h
 * Sequential version of Random Coordinate Descent
 *  Created on: 27-Oct-2015
 *      Author: chandrahas
 */

#ifndef COORDINATEDESCENT_H_
#define COORDINATEDESCENT_H_

#include "Eigen/Dense"

template <class FunctionType>
class CoordinateDescent {
public:
	double threshold;
	double learningRate;
	FunctionType f;
	CoordinateDescent();
	CoordinateDescent(double, double);
	Eigen::VectorXd optimize(Eigen::VectorXd);
	virtual ~CoordinateDescent();
};

#endif /* COORDINATEDESCENT_H_ */
