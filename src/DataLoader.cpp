/*
 * DataLoader.cpp
 * Used for reading data from disk files
 *  Created on: 23-Oct-2015
 *      Author: chandrahas
 */

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <exception>
#include "Eigen/Dense"
#include "ConfigParser/ConfigFile.h"
#include "DataLoader.h"

DataLoader::DataLoader() {
	// Loads the default dataset from the config
	std::string config_file ("config/files.cfg");
	ConfigFile cf(config_file);
	std::string filename = (std::string)cf.Value("DEFAULT", "test");
	const long int M = (int)cf.Value("DEFAULT", "testM");
	const long int N = (int)cf.Value("DEFAULT", "testN");
	const long int W = (int)cf.Value("DEFAULT", "testW");
	this-> M = M;
	this-> N = N;
	this-> omega = W;
	//Bias term is read as the first column
	this->X = Eigen::MatrixXd(M, N+1);
	this->Y = Eigen::VectorXd(M);
	this->rowomega = Eigen::VectorXd(M);
	DataLoader::csvReader(filename, this->M, (this->N + 1));
}

DataLoader::DataLoader(const std::string objective, const std::string dataset) {
	// Loads the default dataset from the config
	std::string config_file ("config/files.cfg");
	ConfigFile cf(config_file);

	std::string filename = (std::string)cf.Value(objective, dataset);
	const long int M = (int)cf.Value(objective, dataset+"M");
	const long int N = (int)cf.Value(objective, dataset+"N");
	const long int W = (int)cf.Value(objective, dataset+"W");
	/*
	std::string filename = (std::string)cf.Value("LASSO", "msd");
	const int M = (int)cf.Value("LASSO", "msdM");
	const int N = (int)cf.Value("LASSO", "msdN");
	*/
	this-> M = M;
	this-> N = N;
	this-> omega = W;
	//Bias term is read as the first column
	this->X = Eigen::MatrixXd(M, N+1);
	this->Y = Eigen::VectorXd(M);
	this->rowomega = Eigen::VectorXd(M);
	DataLoader::csvReader(filename, this->M, (this->N + 1));
}

void DataLoader::csvReader(std::string filename, long int M, long int N){
	std::ifstream fin(filename.c_str());
	try{
		std::string line;
		std::string word;
		int i=0;
		int j=0;
		int nonzero=0;
		double val=0.0;
		while(std::getline(fin, line)){
			//buffer.str(line);
			std::istringstream buffer(line);
			//First column is the label
			std::getline(buffer, word, ',');
			this->Y(i) = atof(word.c_str());
			//std::cout << word;
			j = 0;
			nonzero = 0;
			while(std::getline(buffer, word, ',')){
				//std::cout << word << " ";
				val = atof(word.c_str());
				this->X(i,j) = val;
				nonzero += (val!=0);
				//std::cout << word;
				j++;
			}
			this->rowomega(i) = nonzero;
			i++;
			//std::cout << std::endl;
			//buffer.clear();
		}
		std::cout << "Data Loaded Successfully" << std::endl ;
	}
	catch(std::exception& e){
		std::cerr << "ERROR : " << e.what() << std::endl;
	}
	fin.close();
}

DataLoader::~DataLoader() {
	// TODO Auto-generated destructor stub
	//free(&(this->X));
	//free(&(this->Y));
}
