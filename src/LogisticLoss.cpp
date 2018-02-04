/*
 * LogisticLoss.cpp
 *
 *  Created on: 13-Nov-2015
 *      Author: chandrahas
 */

#include "LogisticLoss.h"

LogisticLoss::LogisticLoss() {
	int m=100;
	int n=10;
	this->A = Eigen::MatrixXd::Random(m,n);
	this->Y = Eigen::MatrixXd::Random(m,1);
	for(int i=0;i<m;i++)
		this->Y(i) = (this->Y(i)) > 0.5 ? 1 : -1 ;
}

LogisticLoss::LogisticLoss(Eigen::MatrixXd x, Eigen::MatrixXd y, long int w){
	this->A = x;
	this->Y = y;
	this->m = x.innerSize();
	this->n = x.outerSize();
	this->regularizer = 0.0;
	if(w <= 0){
		this->omega = (this->A).outerSize();
	}
	else{
		this->omega = w;
	}
}

double LogisticLoss::eval(Eigen::VectorXd x, Eigen::VectorXd Ax){
	int m = this->m;
	int n = this->n;
	Eigen::VectorXd val = Ax; // + b*Eigen::VectorXd::Ones(m); 		//Ax+b
	val = -1 * (val.cwiseProduct(this->Y));							//-Y.*(Ax+b)
	//val = Eigen::VectorXd::Ones(m) + val.array().exp().matrix();	//1+exp(-Y.*(Ax+b))
	val = (1+val.array().exp());
	val = val.array().log();										//log(1+exp(-Y.*(Ax+b)))
	return val.sum()/double(m);										//Sum(log(1+exp(-Y.*(Ax+b))))
}

Eigen::VectorXd LogisticLoss::grad(Eigen::VectorXd x, Eigen::VectorXd Ax){
	int m = this->m;
	int n = this->n;
	Eigen::VectorXd val = Ax; // + b*Eigen::VectorXd::Ones(m); 		//Ax+b
	val = val.cwiseProduct(this->Y);								//Y.*(Ax+b)
	//val = Eigen::VectorXd::Ones(m) + val.array().exp().matrix();	//1+exp(Y.*(Ax+b))
	//val = val.inverse();											//1/(1+exp(Y.*(Ax+b)))
	val = (1+val.array().exp()).inverse();
	val = -1*val.cwiseProduct(this->Y);								//(1/(1+exp(Y.*(Ax+b)))).*(-Y)
	return (1/double(m))*(this->A.transpose())*val;					//(1/(1+exp(Y.*(Ax+b)))).*(-A'Y)
}

double LogisticLoss::grad_i(Eigen::VectorXd x, Eigen::VectorXd Ax, long int i){
	int m = this->m;
	int n = this->n;
	Eigen::VectorXd val = Ax; // + b*Eigen::VectorXd::Ones(m); 		//Ax+b
	val = val.cwiseProduct(this->Y);								//Y.*(Ax+b)
	//val = Eigen::VectorXd::Ones(m) + val.array().exp().matrix();	//1+exp(-Y.*(Ax+b))
	//val = val.inverse();											//1/(1+exp(-Y.*(Ax+b)))
	val = (1+val.array().exp()).inverse();
	val = -1*val.cwiseProduct(this->Y);								//(1/(1+exp(-Y.*(Ax+b)))).*(-Y)
	return (1/double(m))*(this->A.col(i).transpose())*val;			//(1/(1+exp(-Y.*(Ax+b)))).*(-A'Y)
}

double LogisticLoss::grad_b(Eigen::VectorXd x, Eigen::VectorXd Ax){
	int m = this->m;
	int n = this->n;
	Eigen::VectorXd val = Ax; // + b*Eigen::VectorXd::Ones(m); 		//Ax+b
	val = val.cwiseProduct(this->Y);								//Y.*(Ax+b)
	//val = Eigen::VectorXd::Ones(m) + val.array().exp().matrix();	//1+exp(-Y.*(Ax+b))
	//val = val.inverse();											//1/(1+exp(-Y.*(Ax+b)))
	val = (1+val.array().exp()).inverse();
	val = -1*val.cwiseProduct(this->Y);								//(1/(1+exp(-Y.*(Ax+b)))).*(-Y)
	return val.sum()/(double(m));									//(1/(1+exp(-Y.*(Ax+b)))).*(-A'Y)
}

LogisticLoss::~LogisticLoss() {
}
