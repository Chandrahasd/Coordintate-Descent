/*
 * CoordinateDescent.cpp
 *
 *  Created on: 27-Oct-2015
 *      Author: chandrahas
 */

#include "CoordinateDescent.h"
#include "utils.h"
#include <iostream>

#include <chrono>
#include <thread>
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::time_point;
using std::chrono::high_resolution_clock;

template <class FunctionType>
CoordinateDescent<FunctionType>::CoordinateDescent() {
	// TODO Auto-generated constructor stub
	this->learningRate = 0.1;
	this->threshold = 0.1;
}

template <class FunctionType>
CoordinateDescent<FunctionType>::CoordinateDescent(double alpha, double  epsilon) {
	// TODO Auto-generated constructor stub
	this->learningRate = alpha;
	this->threshold = epsilon;
}

template <class FunctionType>
Eigen::VectorXd CoordinateDescent<FunctionType>::optimize(Eigen::VectorXd x0){
	time_point<high_resolution_clock> st,et;
	int i; //current selected index
	int T = 800; //max number of iterations
	int k = 0; //iteration counter
	double g = 0.0; //gradient
	Eigen::VectorXd x = x0;
	Eigen::VectorXd Ax = (this->f.A)*x;
	int m = (this->f.A).innerSize();
	int n = (this->f.A).outerSize();
	Eigen::VectorXd L(n);
	for(i=1;i<n;i++){
		L(i) = (this->f.A).col(i).norm();
		//std::cout << "L(" << i << ") : " << L(i) << std::endl;
	}
	//step lenght for bias term
	L(0) = 1/(this->learningRate);
	st = high_resolution_clock::now();
	while(k<T){
		Ax = (this->f.A)*x;
		i = rand()%n;
		g = this->f.grad_i(x, Ax, i) ;
		//x(i) = x(i) - (this->learningRate)*g ;
		x(i) = x(i) - (1/L(i))*g ;
		k++;
		//if(k%n == 0)
		//std::cout << i << " : " << this->f.grad(Ax).norm() << std::endl ;
		//std::cout << this->f.grad(Ax).norm() << std::endl ;
	}
	et = high_resolution_clock::now();
	std::cout << "Total Time : " << duration_cast<milliseconds>(et-st).count() << std::endl ;

	return x;
}

template <class FunctionType>
CoordinateDescent<FunctionType>::~CoordinateDescent() {
	// TODO Auto-generated destructor stub
}
