/*
 * AcceleratedParallelCoordinateDescent.h
 * Implements the Accelerated Parallel Coordinate Descent method as given by Olivier Fercoq, Peter Richtarik in Accelerated, Parallel and Proximal Coordinate Descent
 *  Created on: 04-Dec-2015
 *      Author: chandrahas
 */

#ifndef ACCELERATEDPARALLELCOORDINATEDESCENT_H_
#define ACCELERATEDPARALLELCOORDINATEDESCENT_H_


#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "DataLoader.h"

template <class FunctionType>
class AcceleratedParallelCoordinateDescent {
public:
	double threshold; //stopping criterion
	int nIterations; //number of iterations
	int period; //iteration interval to reduce learning rate
	double learningRate; // learning rate for bias parameter
 	bool useGSrule; //flag to use GS rule
	std::string outfile; //file to save variable values
	std::ofstream fout; // log file
 	int P;	//number of threads
	Eigen::VectorXd omega; //degree of partial separability for each row
	Eigen::VectorXd stepsize; //columnwise stepsize
	Eigen::VectorXd beta; //columnwise beta values
	std::string regtype,losstype; // loss and regularization types
	FunctionType f; //objective function (loss+regularizer)
	AcceleratedParallelCoordinateDescent();
	AcceleratedParallelCoordinateDescent(double, double, int, int, int, int, FunctionType, DataLoader, const std::string);
	void init();
	void dumpParameters();
	void findUpdate(Eigen::VectorXd, Eigen::VectorXd, double*, int, double);
	Eigen::VectorXd optimize(Eigen::VectorXd);
	void dumpVectors(std::vector<Eigen::VectorXd>);
	void generateTrace(std::string, int=100);
	virtual ~AcceleratedParallelCoordinateDescent();
};

#endif /* ACCELERATEDPARALLELCOORDINATEDESCENT_H_ */
