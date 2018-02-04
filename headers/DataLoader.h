/*
 * DataLoader.h
 * Loads the data from the disk
 *  Created on: 23-Oct-2015
 *      Author: chandrahas
 */

#ifndef DATALOADER_H_
#define DATALOADER_H_

#include <cstring>
#include <iostream>
#include "Eigen/Dense"

class DataLoader {
private:
	std::string config_file();
public:
	Eigen::MatrixXd X;
	Eigen::VectorXd Y;
	long int M;
	long int N;
	long int omega;
	Eigen::VectorXd rowomega;
	DataLoader();
	DataLoader(std::string, std::string);
	void csvReader(std::string, long int, long int);
	virtual ~DataLoader();
};

#endif /* DATALOADER_H_ */
