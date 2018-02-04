/*
 * utils.cpp
 *
 *  Created on: 26-Oct-2015
 *      Author: chandrahas
 */

#include "utils.h"
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cstdlib>
#include <cassert>
#include <map>

#include "Eigen/Dense"

template<class iter>
iter random_shuffle(iter begin, iter end, int m){
	int left = std::distance(begin, end);
	assert(m <= left);
	while(m--){
		iter right = begin;
		std::advance(right, rand()%left);
		std::swap(*begin, *right);
		++begin;
		--left;
	}
	return begin;
}

void niceSampling(int tau, int n, std::vector<int> *a){
	assert(tau <= n);
    for(int i=0; i<n; ++i)
        (*a)[i] = i;
    random_shuffle((*a).begin(), (*a).end(), tau);
}

void sortOnIndices(const Eigen::VectorXd v, std::vector<int> *idx, bool reverse=false) {

  for (size_t i = 0; i != idx->size(); ++i) (*idx)[i] = i;

  if(reverse)
      sort(idx->begin(), idx->end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
  else
      sort(idx->begin(), idx->end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
}


void applyGSrule(int tau, int n, std::vector<int> *a, Eigen::VectorXd grad){
	assert(tau <= n);
    for(int i=0; i<n; ++i)
        (*a)[i] = i;
    grad = grad.cwiseAbs();
    std::sort((*a).begin(), (*a).end(), [&grad](size_t i1, size_t i2){ return grad(i1) > grad(i2); });
	//std::sort(, grad.data()+grad.innerSize(), [&grad](size_t i1, size_t i2){ return grad(i1) > grad(i2); });
}

void normalizeColumn(Eigen::VectorXd *X){
	int n = X->innerSize();
	double mean = X->mean();
	double var = 0;
	double stddev = 0;
	for(int i=0;i<n;i++){
		(*X)(i) = (*X)(i)-mean;
		var += (*X)(i)*(*X)(i);
	}
	stddev = sqrt(var);
	for(int i=0;i<n;i++){
		(*X)(i) /= stddev;
	}

}

void normalizeMatrix(Eigen::MatrixXd *X, int stcol, int etcol){
	int n = X->outerSize();
	int m = X->innerSize();
	if(etcol==0)
		etcol = n;
	for(int j=stcol;j<etcol;j++){
		double mean = X->col(j).mean();
		double var = 0;
		double stddev = 0;
		for(int i=0;i<m;i++){
			(*X)(i,j) = (*X)(i,j)-mean;
			var += (*X)(i,j)*(*X)(i,j);
		}
		stddev = sqrt(var);
		if(stddev == 0)
			continue;
		for(int i=0;i<m;i++){
			(*X)(i,j) /= stddev;
		}
	}
}

int getDOPS(Eigen::MatrixXd X){
	int m = X.innerSize();
	int n = X.outerSize();
	int maxDOPS = -1;
	int curDOPS = 0;
	for(int i=0;i<m;i++){
		curDOPS = 0;
		for(int j=0;j<n;j++){
			if(X(i,j) != 0.0)
				curDOPS++;
		}
		if(curDOPS > maxDOPS)
			maxDOPS = curDOPS;
	}
	return maxDOPS;
}

int find_arg(std::string str, int argc, char **argv) {
    int i;
    for (i = 1; i < argc; i++) {
        if(!strcmp(str.c_str(), argv[i])) {
            if (i == argc - 1) {
                std::cout << "No argument given for" << str << std::endl ;
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

/*
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
*/

/*
int main(){
	size_t n = 3;
	size_t k = 2;
	size_t N = 10000;
	std::vector<int> a(n);
	std::map<int,int> count;
	for(int i=0;i<n;i++)
		count[i] = 0;
	for(int i=0;i<N;i++){
		niceSampling(k, n, &a);
		for(int j=0;j<k;j++){
			//std::cout << a[j] << std::ends;
			count[a[j]] += 1;
		}
		//std::cout << std::endl;
	}
	for(int i=0;i<n;i++)
		std::cout << count[i] << std::endl;
	return 0;
}
*/
