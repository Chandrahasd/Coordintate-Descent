This is an implementation of Coordinate Descent Algorithms for optimization in C++ done as part of my course project. Following algorithms are implemented.
* Coordinate Descent
* [Parallel Coordinate Descent](https://arxiv.org/abs/1212.0873)
* [Accelerated, Parallel and Proximal Coordinate Descent](https://arxiv.org/abs/1312.5799)

Compilation:
---
```shell
$make clean
$make all
```

Running:
---
Use following format
```shell
./CD -o1 option1 -o2 option2 -o3 option3 .... 
```
Options:
* `-A` optimization algorithms. (required)
      * pcd: parallel coordinate descent
      * apcd: accelerated parallel coordinate descent
* `-L` Loss function. (required)
     * sq: squared loss
     * log: logistic loss
* `-R` Regularizer.
     * l1: L1 regularizer
     * l2: L2 regularizer
* `-D` Dataset name. (required) Files are mentioned in config/files.cfg. To add new dataset, please add it in config/files.cfg with details.
     * blog : [Blog Feedback Data Set](http://archive.ics.uci.edu/ml/datasets/BlogFeedback)
     * msd : [Year Prediction MSD](http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD)
     * arcene : [Arcene Data Set](http://archive.ics.uci.edu/ml/datasets/Arcene)
     * internetAd : [Internet Advertisements Data Set](http://archive.ics.uci.edu/ml/datasets/Internet+Advertisements)
     * theoremProving : [First-order theorem proving Data Set](http://archive.ics.uci.edu/ml/datasets/First-order+theorem+proving)
* `-T` number of iterations. (required)
* `-alpha`  learning rate (required)
* `-period` Iterations to reduce learning rate ( use -1 for constant learning rate)
* `-P`      Number of threads. (required)
* `-lambda` Regularization Constant
* `-GS`      1 for using Gauss-Southwell rule else 0


External Libraries Used:
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) : Used for matrix operations
* [ConfigParser](http://www.adp-gmbh.ch/cpp/config_file.html) : Used for reading config files
