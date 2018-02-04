/*
 * ParallelCoordinateDescent.h
 * Implements the parallel version of Coordinate Descent Algorithm as proposed by Peter Richtarik, Martin Takac (2013) in Parallel Coordinate Descent Methods for Big data Optimization
 *  Created on: 28-Oct-2015
 *      Author: chandrahas
 */

#ifndef PARALLELCOORDINATEDESCENT_H_
#define PARALLELCOORDINATEDESCENT_H_

#include <iostream>
#include <fstream>
#include "Eigen/Dense"

template <class FunctionType>
class ParallelCoordinateDescent {
public:
	double threshold; // stopping criterion
	int nIterations; //number of iterations
	int period; //iteration interval to reduce learning rate
	double learningRate; // learning rate for bias term
	std::string outfile; // file to save variables
	std::ofstream fout; // log file
	bool useGSrule; // flag to use GS rule
	int P;	//number of threads
	int omega; //degree of partial separability (max)
	double beta; // step-size modifiers
	std::string regtype,losstype; // type of loss and regularizer
	Eigen::VectorXd L, w; // Lipschitz contants and update weights
	FunctionType f; // Objective function
	ParallelCoordinateDescent();
	ParallelCoordinateDescent(double, double, int, int, int, int, FunctionType, const std::string);
	void init();
	void dumpParameters();
	void findUpdate(Eigen::VectorXd, Eigen::VectorXd, double*, int);
	Eigen::VectorXd optimize(Eigen::VectorXd);
	void dumpVectors(std::vector<Eigen::VectorXd>);
	void generateTrace(std::string, int=100);
	virtual ~ParallelCoordinateDescent();
};

#endif /* PARALLELCOORDINATEDESCENT_H_ */
