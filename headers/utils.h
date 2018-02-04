/*
 * utils.h
 *
 *  Created on: 26-Oct-2015
 *      Author: chandrahas
 */

#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include "Eigen/Dense"

class utils {
};

void niceSampling(int, int, std::vector<int>*);
void applyGSrule(int, int, std::vector<int>*, Eigen::VectorXd);
void normalizeMatrix(Eigen::MatrixXd*, int stcol=0, int etcol=0);
void normalizeColumn(Eigen::VectorXd*);
int getDOPS(Eigen::MatrixXd);
int find_arg(std::string str, int argc, char **argv);

template <class FunctionType>
void generateTrace(std::string filename, FunctionType f){
	int T,n;
	Eigen::VectorXd x;
	std::ifstream fin(filename, std::ifstream::binary);
	if(fin.fail()){
		std::cout << "Error in reading file. Exitting..." << std::endl ;
		return ;
	}
	Eigen::VectorXd Ax;
	fin.read((char*)&T, sizeof(T));
	fin.read((char*)&n, sizeof(n));
	std::cout << T << " " << n << std::endl ;
	//for(it=xlist.begin(); it<xlist.end(); it++){
	x.resize(n);
	for(int i=0;i<T;i++){
		fin.read((char*)x.data(), sizeof(double)*n);
		Ax = (f.A)*x;
		std::cout << i << " " << (f.eval(Ax))/(10.0e+08) << " " << f.grad(Ax).norm() << std::endl ;
	}
	fin.close();
}

#endif /* UTILS_H_ */
