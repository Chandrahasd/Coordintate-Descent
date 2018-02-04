/*
 * ParallelCoordinateDescent.cpp
 * Implements the parallel version of Coordinate Descent Algorithm as proposed by Peter Richtarik, Martin Takac (2013) in Parallel Coordinate Descent Methods for Big data Optimization
 *  Created on: 28-Oct-2015
 *      Author: chandrahas
 */

#include <iostream>
#include <fstream>
#include <thread>
#include <cstring>
#include "ParallelCoordinateDescent.h"
#include "Eigen/Dense"
#include "utils.h"

#include <chrono>
using std::chrono::milliseconds;
using std::chrono::duration_cast;
using std::chrono::time_point;
using std::chrono::high_resolution_clock;

template <class FunctionType>
ParallelCoordinateDescent<FunctionType>::ParallelCoordinateDescent() {
	// Unused Constructor

}

template <class FunctionType>
ParallelCoordinateDescent<FunctionType>::ParallelCoordinateDescent(double alpha, double epsilon, int p, int T, int per, int gs, FunctionType ff, const std::string filename){
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
	this->period = per;
	this->P = p;
	this->f = ff;
	this->regtype = "na";
	this->losstype = "sq";
}

//save the parameters into the log
template <class FunctionType>
void ParallelCoordinateDescent<FunctionType>::dumpParameters(){
	this->fout << "Method\t:\tParallel Coordinate Descent" << std::endl;
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

//initialize parameters
template <class FunctionType>
void ParallelCoordinateDescent<FunctionType>::init(){
	int m = (this->f.A).innerSize();
	int n = (this->f.A).outerSize();
	this->L = Eigen::VectorXd(n);
	this->w = Eigen::VectorXd(n);
	//using P-nice sampling
	this->omega = this->f.omega ; //n;
	//omega excludes bias term
	this->beta = 1 + (this->omega - 1)*(this->P - 1)/double(std::max<int>(1,n-2));  //min(this->omega, this->P);
	//initialize Lipschitz constants
	for(int i=1;i<n;i++){
		this->L(i) = pow((this->f.A).col(i).norm(),2) ;
		if(L(i) == 0.0)
			L(i) = 1.0;
		this->w(i) = L(i);
	}
	//this->L(0) = 0.5/((this->beta)*(this->learningRate));
	this->L(0) = 1/(this->learningRate);
	this->w(0) = this->L(0);
}

//It finds the update for every thread in parallel
template <class FunctionType>
void ParallelCoordinateDescent<FunctionType>::findUpdate(Eigen::VectorXd x, Eigen::VectorXd Axt, double* update, int idx){
	if(idx<0 || (idx>= (this->f.A).outerSize())){
		this->fout << "idx not in range" << std::endl;
		return;
	}
	double grad = this->f.grad_i(x, Axt, idx) ;
	double denom = (this->beta)*(this->w[idx]);
	if(this->regtype.compare("l2") == 0)
		denom = denom + this->f.regularizer;
	//un-comment following update to use exact gradient descent
	//*update = -grad/(this->w[idx]);
	//following update is proposed in the paper Mentioned at the top of the code
	*update = -grad/denom;

}

//Main routine to optimize function value
template <class FunctionType>
Eigen::VectorXd ParallelCoordinateDescent<FunctionType>::optimize(Eigen::VectorXd x0){
	//dump the final parameter values
	this->dumpParameters();
	std::cout << "Starting Optimization" << std::endl ;
	std::thread t[this->P];
	time_point<high_resolution_clock> st,et;
	milliseconds total_time = duration_cast<milliseconds>(st-st); // 0ms
	//initial variable values
	Eigen::VectorXd x = x0;
	//for writing values of x into file
	std::ofstream outf(this->outfile, std::ofstream::binary);

	int m = (this->f.A).innerSize();
	int n = (this->f.A).outerSize();
	int T = this->nIterations; //Number of iterations
	int BiasPeriod = T;
	std::vector<int> a(n);
	// updates in variable
	double update[this->P];
	// compute Ax in advance to save time
	Eigen::VectorXd Ax;
	Ax = (this->f.A)*x;
	Eigen::VectorXd g ; //= (this->f).grad(x, Ax);
	double fval = 0.0;
	double tmp=0.0;
	int k=0;
	// save params to file
	outf.write((char*)&T, sizeof(T));
	outf.write((char*)&n, sizeof(n));
	outf.write((char*)(&(this->P)), sizeof(this->P));
	while((k++)<T){
		st = high_resolution_clock::now();
		//std::cout << "Iteration : " << k << std::endl;
		// select the coordinates using either GS-Rule or nice-sampling
		if(this->useGSrule){
			g = (this->f).grad(x, Ax);
			applyGSrule(this->P, n, &a, g);
		}
		else{
			niceSampling(this->P, n, &a);
		}
		//Spawn threads
		for(int i=0;i<(this->P);i++){
			t[i] = std::thread(&ParallelCoordinateDescent<FunctionType>::findUpdate, this, x, Ax, &update[i], a[i]);
		}
		//wait for synchronization
		for(int i=0;i<(this->P);i++){
			t[i].join();
		}
		//apply the updates
		for(int i=0;i<(this->P);i++){
			tmp = x[a[i]] + update[i] ;
			if( (tmp*x[a[i]] < 0) && ((this->regtype).compare("l1")==0) )
				x[a[i]] = 0;
			else
				x[a[i]] = tmp;
		}
		Ax = (this->f.A)*x;
		et = high_resolution_clock::now();
		//decrease learning rate for bias term after BiasPeriod iterations
		this->w(0)  = ((k+1)%BiasPeriod == 0) ? 2*(this->w(0)) : this->w(0);
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
void ParallelCoordinateDescent<FunctionType>::dumpVectors(std::vector<Eigen::VectorXd> xlist){
	std::ofstream outf("testx.bin", std::ofstream::binary);
	outf.write((char*)&xlist[0], xlist.size()*sizeof(xlist));
	outf.close();
}

//Calculates function values using the variable values generated by algorithm
template <class FunctionType>
void ParallelCoordinateDescent<FunctionType>::generateTrace(std::string filename, int interval){
	int T,n,p;
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
	fin.read((char*)&p, sizeof(p));
	//this->fout << T << " " << n << std::endl ;
	this->fout << "T : " << T << std::endl ;
	this->fout << "n : " << n << std::endl ;
	this->fout << "P : " << p << std::endl ;
	//for(it=xlist.begin(); it<xlist.end(); it++){
	std::vector<int> a(p);
	x.resize(n);
	for(int i=0;i<T;i++){
		for(int j=0; j<p;j++)
			fin.read((char*)&a[j], sizeof(a[0]));
		fin.read((char*)x.data(), sizeof(double)*n);
		if(i%interval==0){
			Ax = (this->f.A)*x;
			this->fout << "Idxs : ";
			for(int j=0;j<p;j++)
				this->fout << a[j] << " " ;
			this->fout << std::endl;
			this->fout << "Trace: " << i << " " << ((this->f).eval(x, Ax)) << " " << (this->f).grad(x, Ax).norm() << " " << (this->f).grad_i(x, Ax, 0)<< std::endl ;
		}
	}
	fin.close();
}

template <class FunctionType>
ParallelCoordinateDescent<FunctionType>::~ParallelCoordinateDescent() {
	this->fout.close();
}
