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
* `-D` Dataset name. (required)
    * blog, msd, arcene etc. (Specified in config/files.cfg. To add new dataset, please add it in config/files.cfg with details)
* `-T` number of iterations. (required)
* `-alpha`  learning rate (required)
* `-period` Iterations to reduce learning rate ( use -1 for constant learning rate)
* `-P`      Number of threads. (required)
* `-lambda` Regularization Constant
* `-GS`      1 for using Gauss-Southwell rule else 0


External Libraries Used:
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) : Used for matrix operations
* [ConfigParser](http://www.adp-gmbh.ch/cpp/config_file.html) : Used for reading config files
