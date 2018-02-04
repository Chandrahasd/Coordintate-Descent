/*
 * main.cpp
 * Main file used to invoke given algorithms
 *
 *  Created on: 28-Oct-2015
 *      Author: chandrahas
 */

#include <iostream>
#include <fstream>
#include "Eigen/Dense"
#include "QuadraticForm.h"
#include "SteepestGradientDescent.h"
#include "DataLoader.h"
#include "SquaredLoss.h"
#include "LogisticLoss.h"
#include "CoordinateDescent.h"
#include "CoordinateDescent.cpp"
#include "ParallelCoordinateDescent.h"
#include "ParallelCoordinateDescent.cpp"
#include "AcceleratedParallelCoordinateDescent.h"
#include "AcceleratedParallelCoordinateDescent.cpp"
#include "L2Regularizer.h"
#include "L2Regularizer.cpp"
#include "L1Regularizer.h"
#include "L1Regularizer.cpp"
#include "utils.h"
//#include "utils.cpp"


using namespace std;
using Eigen::MatrixXd;
using Eigen::Matrix;

void testSGD(){
    MatrixXd Q(2,2);
    Q << 1,1,1,1;
    VectorXd b(2);
    b << 0,0 ;
    QuadraticForm *q = new QuadraticForm(Q,b);
    int n = 2;
    VectorXd x = VectorXd::Random(n);
    std::cout <<"Function val is : " << q->eval(x) << std::endl;
    std::cout <<"Gradient val is : " << q->grad(x) << std::endl;
    std::cout <<"Hessian val is : " << q->hessian(x) << std::endl;
    SteepestGradientDescent *sgd = new SteepestGradientDescent(0.1, 0.0001, q);
    sgd->optimize();
    delete(sgd);
    delete(q);
}

void testDataLoader(){
	DataLoader dl = DataLoader();
	//MatrixXd A(100,91);
	dl.csvReader("../datasets/testdataset.csv", 100, 91);
	std::cout << dl.X;
	std::cout << dl.Y;
}

void testSquaredLoss(){
	MatrixXd A(2,2);
	VectorXd Y(2);
	VectorXd x(2);
	A << 1,2,2,1;
	Y << 2,3;
	x << 2,2;
	SquaredLoss *s = new SquaredLoss(A, Y);
	std::cout << "Function value at : " << x << " is " << s->eval(x, A*x) << std::endl;
	std::cout << "Function grad  at : " << x << " is " << s->grad(x, A*x) << std::endl;
	std::cout << "Gradient wrt 0 at : " << x << " is " << s->grad_i(x, A*x,0) << std::endl;
	std::cout << "Gradient wrt 1 at : " << x << " is " << s->grad_i(x, A*x,1) << std::endl;
	std::cout << "Gradient wrt b at : " << x << " is " << s->grad_b(x, A*x) << std::endl;
	delete s;
}

void testCoordinateDescent(){
	CoordinateDescent<SquaredLoss> cd(0.1,0.1);
	MatrixXd A(2,2);
	VectorXd Y(2);
	VectorXd x(2);
	A << 1,2,2,1;
	Y << 2,3;
	x << 2,2;
	cd.f = SquaredLoss(A, Y);
	std::cout << cd.optimize(VectorXd::Random(2));
}

void testParallelCD(double alpha, long int T, int P, double lambda, std::string ds, std::string loss, std::string reg="l2"){
	DataLoader dl = DataLoader(loss, ds);
	int m = dl.M;
	int n = dl.N + 1;
	VectorXd x = VectorXd::Random(n);
	normalizeMatrix(&(dl.X),1);
	L2Regularizer<SquaredLoss> f(dl.X, dl.Y, dl.omega, lambda);
	ParallelCoordinateDescent<L2Regularizer<SquaredLoss>> cd(alpha, 0.1,P, T, T, 0, f,"pcd_"+ds+"_"+loss+"_"+reg);
	cd.regtype = reg;
	cd.init();
	x = cd.optimize(x);
	std::string filename = cd.outfile;
	cd.generateTrace(filename);
}

template <class FunctionType>
void runPCD(double alpha, long int T, int P, int per, int GS, FunctionType f, Eigen::VectorXd x, std::string ds, std::string loss, std::string reg="l2"){
	ParallelCoordinateDescent<FunctionType> cd(alpha, 0.1, P, T, per, GS, f,"./vectors/pcd_"+ds+"_"+loss+"_"+reg);
	cd.regtype = reg;
	cd.init();
	x = cd.optimize(x);
	cd.generateTrace(cd.outfile, ((T<=1000)?1:T/100));
}

template <class FunctionType>
void runAPCD(double alpha, long int T, int P, int per, int GS, FunctionType f, DataLoader dl, Eigen::VectorXd x, std::string ds, std::string loss, std::string reg="l2"){
	AcceleratedParallelCoordinateDescent<FunctionType> cd(alpha, 0.1, P, T, per, GS, f, dl, "./vectors/apcd_"+ds+"_"+loss+"_"+reg);
 	cd.regtype = reg;
	cd.init();
	x = cd.optimize(x);
	cd.generateTrace(cd.outfile, ((T<=1000)?1:T/100));
}

void runAlgorithm(std::string algo, double alpha, long int T, int P, int per, int GS, double lambda, std::string ds, std::string loss, std::string reg="l2"){
	DataLoader dl = DataLoader(loss, ds);
	int m = dl.M;
	int n = dl.N + 1;
	VectorXd x = VectorXd::Random(n);
	normalizeMatrix(&(dl.X),1);
	if(algo.compare("pcd")==0){
		if(loss.compare("sq")==0){
			if(lambda == 0.0){
				SquaredLoss f(dl.X, dl.Y, dl.omega);
				runPCD<SquaredLoss>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
			else if(reg.compare("l1")==0){
				L1Regularizer<SquaredLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runPCD<L1Regularizer<SquaredLoss>>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
			else if(reg.compare("l2")==0){
				L2Regularizer<SquaredLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runPCD<L2Regularizer<SquaredLoss>>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
		}
		else if(loss.compare("log")==0){
			if(lambda == 0.0){
				LogisticLoss f(dl.X, dl.Y, dl.omega);
				runPCD<LogisticLoss>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
			else if(reg.compare("l1")==0){
				L1Regularizer<LogisticLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runPCD<L1Regularizer<LogisticLoss>>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
			else if(reg.compare("l2")==0){
				L2Regularizer<LogisticLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runPCD<L2Regularizer<LogisticLoss>>(alpha, T, P, per, GS, f, x, ds, loss, reg);
			}
		}
	}
	else{
		if(loss.compare("sq")==0){
			if(lambda == 0.0){
				SquaredLoss f(dl.X, dl.Y, dl.omega);
				runAPCD<SquaredLoss>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
			else if(reg.compare("l1")==0){
				L1Regularizer<SquaredLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runAPCD<L1Regularizer<SquaredLoss>>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
			else if(reg.compare("l2")==0){
				L2Regularizer<SquaredLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runAPCD<L2Regularizer<SquaredLoss>>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
		}
		else if(loss.compare("log")==0){
			if(lambda == 0.0){
				LogisticLoss f(dl.X, dl.Y, dl.omega);
				runAPCD<LogisticLoss>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
			else if(reg.compare("l1")==0){
				L1Regularizer<LogisticLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runAPCD<L1Regularizer<LogisticLoss>>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
			else if(reg.compare("l2")==0){
				L2Regularizer<LogisticLoss> f(dl.X, dl.Y, dl.omega, lambda);
				runAPCD<L2Regularizer<LogisticLoss>>(alpha, T, P, per, GS, f, dl, x, ds, loss, reg);
			}
		}
	}
}

void testMatrix(){
	int m=3;
	int n=4;
	Eigen::MatrixXd A = Eigen::MatrixXd::Random(m,n);
	//A << 1,2,2,1,1,1,2,2,1,2,2,1,1,2,2,1;
	std::cout << "Size of matrix A is " << A.innerSize() << std::endl;
	std::cout << "Size of matrix A is " << A.outerSize() << std::endl;
	std::cout << "A : " << A << std::endl;
	Eigen::MatrixXd B = A.block(0, 1, m, n-1);
	std::cout << "Block : " << B;
	B = A.block(0, 0, m, 1);
	std::cout << "Block : " << B;
}

void testUtils(){
	Eigen::MatrixXd x(2,5);
	x << 1,2,3,4,5,6,7,8,9,0;
	std::cout << x << std::endl ;
	normalizeMatrix(&x,1,3);
	std::cout << x << std::endl ;
}

void testDOPS(){
	DataLoader dl = DataLoader("LASSO", "msd");
	int dops = getDOPS(dl.X);
	std::cout << "MSD Dimension : " << dl.X.innerSize() << " X "<< dl.X.outerSize() << std::endl;
	std::cout << "MSD DOPS : " << dops << std::endl ;
	dl = DataLoader("LASSO", "blog");
	dops = getDOPS(dl.X);
	std::cout << "BLOG Dimension : " << dl.X.innerSize() << " X " << dl.X.outerSize() << std::endl;
	std::cout << "BLOG DOPS : " << dops << std::endl ;
	dl = DataLoader("LOGREG", "arcene");
	dops = getDOPS(dl.X);
	std::cout << "Arcene Dimension : " << dl.X.innerSize() << " X " << dl.X.outerSize() << std::endl;
	std::cout << "Arcene DOPS : " << dops << std::endl ;
	dl = DataLoader("LOGREG", "internetAd");
	dops = getDOPS(dl.X);
	std::cout << "InternetAd Dimension : " << dl.X.innerSize() << " X " << dl.X.outerSize() << std::endl;
	std::cout << "InternetAd DOPS : " << dops << std::endl ;
	dl = DataLoader("LOGREG", "theoremProving");
	dops = getDOPS(dl.X);
	std::cout << "Theorem Dimension : " << dl.X.innerSize() << " X " << dl.X.outerSize() << std::endl;
	std::cout << "Theorem DOPS : " << dops << std::endl ;
}

int main(int argc, char* argv[]) {
   if ( argc == 1){
        std::cout << "Use following format" << std::endl;
        std::cout << "./CD -o1 option1 -o2 option2 -o3 option3 .... " << std::endl ;
        std::cout << "-A\toptimization algorithms. (required)" << std::endl;
        std::cout << "\tValues: pcd: parallel coordinate descent\n\t\tapcd: accelerated parallel coordinate descent" << std::endl;
        std::cout << "-L\tLoss function. (required)" << std::endl;
        std::cout << "\tValues: sq:squared loss\n\t\tlog:logistic loss" << std::endl;
        std::cout << "-R\tRegularizer." << std::endl;
        std::cout << "\tValues: l1: L1 regularizer\n\t\tl2: L2 regularizer" << std::endl;
        //std::cout << "-N\tnormalize data" << std::endl;
        //std::cout << "\tValues: 1:yes\n\t\t0:no" << std::endl;
        std::cout << "-D\tDataset name. (required)" << std::endl;
        std::cout << "\tValues: test, msd, arcene" << std::endl;
        std::cout << "-T\tnumber of iterations. (required)" << std::endl;
        std::cout << "-alpha\tlearning rate (required)" << std::endl;
        std::cout << "-period\tIterations to reduce learning rate ( use -1 for constant learning rate)" << std::endl;
        std::cout << "-P\tNumber of threads. (required)" << std::endl;
        std::cout << "-lambda\tRegularization Constant" << std::endl;
        std::cout << "-GS\t 1 for using Gauss-Southwell rule else 0" << std::endl;
    }
    std::string options[] = {"-A", "-L", "-R", "-N", "-D", "-T", "-alpha", "-P", "-lambda", "-period", "-GS"};
    int len=11;
    int i,j;
    double alpha = -1.0;
    long int T=-1;
    int per = 500;
    int P=-1;
    int N=1;
    int GS = 0;
    double lambda=0;
    std::string reg("l2");
    std::string loss("sq");
    std::string algo("pcd");
    std::string ds("msd");

    for(i=0;i<len;i++){
        j = find_arg(options[i], argc, argv);
        //std::cout << options[i] << " : " << argv[j+1] << std::endl ;
        if(i==0)
        	algo = std::string(argv[j+1]);
        else if(i==1)
        	loss = std::string(argv[j+1]);
        else if(i==2)
        	reg = std::string(argv[j+1]);
        else if(i==3)
        	N = atoi(argv[j+1]);
        else if(i==4)
        	ds = std::string(argv[j+1]);
        else if(i==5)
        	T = atoi(argv[j+1]);
        else if(i==6)
        	alpha = atof(argv[j+1]);
        else if(i==7)
        	P = atoi(argv[j+1]);
        else if(i==8)
        	lambda = atof(argv[j+1]);
        else if(i==9)
        	per = atoi(argv[j+1]);
        else if(i==10)
        	GS = atoi(argv[j+1]);
    }
    if(T<1){
    	std::cout << "Please give iterations" << std::endl;
    }
    if(P<0)
    	P = 1;
    if(alpha < 0.0)
    	alpha = 0.003;
    if(per <= 0)
    	per = T;
    if((reg.compare("l2")!=0)&&(reg.compare("l1") !=0 ))
    	lambda = 0.0;
    //std::cout << "using alpha value : " << alpha << std::endl ;
	std::cout << "Run Details: " << std::endl ;
	std::cout << "Algorithm " << algo << std::endl ;
	std::cout << "Loss " << loss << std::endl ;
	std::cout << "Regularizer " << reg << std::endl ;
	std::cout << "Dataset " << ds << std::endl ;
	std::cout << "Iterations " << T << std::endl ;
	std::cout << "Threads " << P << std::endl ;
	std::cout << "Regularization Constant " << lambda << std::endl ;
	std::cout << "Learning Rate " << alpha << std::endl ;
	std::cout << "Number of iterations to reduce learning rate " << per << std::endl ;
	std::cout << "Use Gauss-Southwell rule " << GS << std::endl ;
	if((T <= 1) || (P < 1) || (alpha < 0.0) || ((algo.compare("pcd")!=0)&&(algo.compare("apcd") !=0 )) || ((loss.compare("log")!=0)&&(loss.compare("sq") !=0 )) ){
		std::cout <<" Please provide all the necessary parameters" << std::endl ;
		return -1;
	}
	runAlgorithm(algo, alpha, T, P, per, GS, lambda, ds, loss, reg);
	std::cout << "Done....." << std::endl ;
	return 0;
}
