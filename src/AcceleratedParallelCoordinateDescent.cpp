/*
 * AcceleratedParallelCoordinateDescent.cpp
 * Implements the Accelerated Parallel Coordinate Descent method as given by Olivier Fercoq, Peter Richtarik in Accelerated, Parallel and Proximal Coordinate Descent
 *
 *  Created on: 04-Dec-2015
 *      Author: chandrahas
 */

#include "AcceleratedParallelCoordinateDescent.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <cstring>
#include <algorithm>
#include "AcceleratedParallelCoordinateDescent.h"
#include "Eigen/Dense"
#include "utils.h"
#include "DataLoader.h"

#include <chrono>
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::time_point;
using std::chrono::high_resolution_clock;

template <class FunctionType>
AcceleratedParallelCoordinateDescent<FunctionType>::AcceleratedParallelCoordinateDescent() {
	// Unused Constructor

}

template <class FunctionType>
AcceleratedParallelCoordinateDescent<FunctionType>::AcceleratedParallelCoordinateDescent(double alpha, double epsilon, int p, int T, int per, int gs, FunctionType ff, DataLoader dl, const std::string filename){
	this->threshold = epsilon;
	this->learningRate = alpha;
	this->nIterations = T;
	if(gs==1){
		this->useGSrule = true;
		this->outfile = filename + "_" + std::to_string(p) + "_" + std::to_string(T) + "_gs.bin";
		(this->fout).open(filename + "_" + std::to_string(p) + "_" + std::to_string(T) + "_gs.log", std::fstream::out);
	}
	else{
		this->useGSrule = false;
		this->outfile = filename + "_" + std::to_string(p) + "_" + std::to_string(T) + ".bin";
		(this->fout).open(filename + "_" + std::to_string(p) + "_" + std::to_string(T) + ".log", std::fstream::out);
	}
	this->P = p;
	this->period = per;
	this->f = ff;
	this->regtype = "na";
	this->losstype = "sq";
	this->omega = dl.rowomega;
}

//save the parameters
template <class FunctionType>
void AcceleratedParallelCoordinateDescent<FunctionType>::dumpParameters(){
	this->fout << "Method\t:\tAccelerated Parallel Coordinate Descent" << std::endl;
	this->fout << "Loss\t:\t" << this->losstype << std::endl;
	this->fout << "Regularizer type\t:\t" << this->regtype << std::endl;
	this->fout << "Iterations\t:\t" << this->nIterations << std::endl;
	this->fout << "Learning Rate\t:\t" << this->learningRate << std::endl;
	this->fout << "Threads\t:\t" << this->P << std::endl;
	this->fout << "Learning Rate Period\t:\t" << this->period  << std::endl;
	if(this->useGSrule)
		this->fout << "Using Gauss-Southwell Rule\t:\t" << "YES" << std::endl ;
	else
		this->fout << "Using Gauss-Southwell Rule\t:\t" << "NO" << std::endl ;
}

template <class FunctionType>
void AcceleratedParallelCoordinateDescent<FunctionType>::init(){
	int m = (this->f.A).innerSize();
	int n = (this->f.A).outerSize();
	this->omega = Eigen::VectorXd(m);
	this->stepsize = Eigen::VectorXd(n);
	this->beta = Eigen::VectorXd(m);
	double Lfi = 1.0;
	if((this->losstype).compare("log"))
		Lfi = 0.25;
	// -2 is used to avoid counting bias term
	this->beta = 1 + (this->P -1)*(this->omega.array() - 2)/(std::max<int>(1,n-2));
	//initialize stepsizes
	for(int i=0;i<n;i++){
		this->stepsize(i) = Lfi*(this->beta).dot(((this->f.A).col(i).cwiseProduct((this->f.A).col(i))));
	}
	this->stepsize(0) = 1.0/(n*(this->learningRate));
	this->fout << "stepsize(0) is : " << this->stepsize(0) << std::endl ;
}

// It finds the update in parallel
template <class FunctionType>
void AcceleratedParallelCoordinateDescent<FunctionType>::findUpdate(Eigen::VectorXd x, Eigen::VectorXd Axt, double* update, int idx, double theta){
	int n = (this->f.A).outerSize();
	if(idx<0 || (idx>= n)){
		this->fout << "idx not in range" << std::endl;
		return;
	}
	double grad = this->f.grad_i(x, Axt, idx) ;
	if(idx==0){
		*update = -(this->learningRate)*grad;
		return;
	}
	double denom = n*theta*(this->stepsize(idx))/(this->P);
	if(this->regtype.compare("l2")==0)
		denom = denom + this->f.regularizer ;
	*update = -grad/denom;
}

//Main routine to optimize function value
template <class FunctionType>
Eigen::VectorXd AcceleratedParallelCoordinateDescent<FunctionType>::optimize(Eigen::VectorXd x0){
	//save the final parameters
	std::cout << "Starting Optimization" << std::endl ;
	this->dumpParameters();
	std::thread t[P];

	time_point<high_resolution_clock> st,et;
	milliseconds total_time = duration_cast<milliseconds>(st-st); // 0ms

	int m = (this->f.A).innerSize();
	int n = (this->f.A).outerSize();
	int T = this->nIterations; //Number of iterations
	int BiasPeriod = this->period;
	double theta = (this->P)/double(n);

	//initial values
	Eigen::VectorXd z = x0;
	Eigen::VectorXd u = Eigen::VectorXd::Zero(n);
	Eigen::VectorXd x = (theta*theta)*u + z;
	Eigen::VectorXd Ax;
	Ax = (this->f.A)*x;
	// initilize data file
	std::ofstream outf(this->outfile, std::ofstream::binary);
	std::vector<int> a(n);
	double update[this->P];
	Eigen::VectorXd g = (this->f).grad(x, Ax);
	this->fout << std::endl;
	double fval = 0.0;
	double tmp=0.0;
	int k=0;
	//save params to file
	outf.write((char*)&T, sizeof(T));
	outf.write((char*)&n, sizeof(n));
	outf.write((char*)&(this->P), sizeof(this->P));
	while((k++)<T){
		st = high_resolution_clock::now();
		// select the coordinates using either GS-Rule or nice-sampling
		if(this->useGSrule){
			g = (this->f).grad(x, Ax);
			applyGSrule(this->P, n, &a, g);
		}
		else{
			niceSampling(this->P, n, &a);
		}
		//spawn threads
		for(int i=0;i<(this->P);i++){
			t[i] = std::thread(&AcceleratedParallelCoordinateDescent<FunctionType>::findUpdate, this, z, Ax, &update[i], a[i], theta);
		}
		// wait to synchronize threads
		for(int i=0;i<(this->P);i++){
			t[i].join();
		}
		//apply the updates
		for(int i=0;i<(this->P);i++){
			tmp = z[a[i]] + update[i] ;
			if( (tmp*z[a[i]] < 0) && ((this->regtype).compare("l1")==0) ){
				update[i] =  -z[a[i]] ;
				z[a[i]] = 0;
			}
			else{
				z[a[i]] = tmp;
			}
			u[a[i]] = u[a[i]] - (1-n*theta/(this->P))*update[i]/(theta*theta);
		}

		theta = 0.5*(sqrt(pow(theta,4)+4*pow(theta,2))- pow(theta,2));
		x = pow(theta,2)*u.array() + z.array();
		Ax = (this->f.A)*x;
		//decrease learning rate for bias term after BiasPeriod iterations
		if((k+1)%BiasPeriod == 0)
			this->learningRate = 0.5*(this->learningRate);
		et = high_resolution_clock::now();
		total_time += std::chrono::duration_cast<std::chrono::milliseconds>(et-st);
		if((k+1)%100 == 0){
			std::cout << (k+1) << " Iterations done" << std::endl;
		}
		// save variables to file
		for(int i=0; i<(this->P); i++)
			outf.write((char*)&a[i], sizeof(a[0]));
		outf.write((char*)x.data(), sizeof(x(0))*n);
	}
	this->fout << "Total Time : " << total_time.count() << std::endl ;
	outf.close();
	return x;
}

template <class FunctionType>
void AcceleratedParallelCoordinateDescent<FunctionType>::dumpVectors(std::vector<Eigen::VectorXd> xlist){
	std::ofstream outf("testx.bin", std::ofstream::binary);
	outf.write((char*)&xlist[0], xlist.size()*sizeof(xlist));
	outf.close();
}

// For evaluating function values at variables values generated by algorithm
template <class FunctionType>
void AcceleratedParallelCoordinateDescent<FunctionType>::generateTrace(std::string filename, int interval){
	int T,n,P;
	Eigen::VectorXd x;
	std::cout << "Calculating function values at interval : " << interval << std::endl ;
	std::ifstream fin(filename, std::ifstream::binary);
	if(fin.fail()){
		this->fout << "Error in reading file. Exitting..." << std::endl ;
		return ;
	}
	Eigen::VectorXd Ax;
	fin.read((char*)&T, sizeof(T));
	fin.read((char*)&n, sizeof(n));
	fin.read((char*)&P, sizeof(P));
	this->fout << T << " " << n << std::endl ;
	std::vector<int> a(P);
	x.resize(n);
	for(int i=0;i<T;i++){
		for(int j=0; j<P;j++)
			fin.read((char*)&a[j], sizeof(a[0]));
		fin.read((char*)x.data(), sizeof(double)*n);
		if(i%interval==0){
			Ax = (this->f.A)*x;
			this->fout << "Idxs : ";
			for(int j=0;j<P;j++)
				this->fout << a[j] << " " ;
			this->fout << std::endl;
			this->fout << "Trace: " << i << " " << ((this->f).eval(x, Ax)) << " " << (this->f).grad(x, Ax).norm() << " " << (this->f).grad_i(x, Ax, 0)<< std::endl ;
		}
	}
	fin.close();
}

template <class FunctionType>
AcceleratedParallelCoordinateDescent<FunctionType>::~AcceleratedParallelCoordinateDescent() {
	this->fout.close();
}
