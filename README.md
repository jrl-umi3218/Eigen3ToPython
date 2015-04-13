[![Build Status](https://travis-ci.org/jorisv/Eigen3ToPython.svg?branch=master)](https://travis-ci.org/jorisv/Eigen3ToPython)

Eigen3ToPython
======

Eigen3ToPython aim to make a bidirectional bridge between Numpy and Eigen3.

This python library goal is not to provide a full Eigen3 python binding but to provide easy conversion between Numpy and Eigen3.

Documentation
------

This library allow to:
 * Make operations on fix size Eigen3 matrix
 * Make operations on dynamic size Eigen3 matrix
 * Convert fix and dynamic size Eigen3 matrix to Numpy matrix (`np.matrix`).
 * Convert Numpy matrix (`np.matrix`) to fix or dynamic size Eigen3 matrix

### Fix size Eigen3 Matrix operations

### Dynamic size Eigen3 Matrix operations

### Converting Eigen3 <=> Numpy


Installing
------

### Dependencies

To compile you need the following tools:
 
 * [Git]()
 * [CMake]() >= 2.8
 * [pkg-config]()
 * [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.2
 * [PyBindGen](https://launchpad.net/pybindgen) = 0.16 (build with 0.17 but a bug
in this version prevent to use len)

### Building

```sh
git clone --recursive https://github.com/jorisv/Eigen3ToPython.git
cd Eigen3ToPython
mkdir _build
cd _build
cmake [options] ..
make && make intall
```

Where the main options are:

 * `-DCMAKE_BUIlD_TYPE=Release` Build in Release mode
 * `-DCMAKE_INSTALL_PREFIX=some/path/to/install` default is `/usr/local`


Pulling git subtree
------

To update sync cmake or .travis directory with their upstream git repository:

	git fetch git://github.com/jrl-umi3218/jrl-cmakemodules.git master
	git subtree pull --prefix cmake git://github.com/jrl-umi3218/jrl-cmakemodules.git master --squash

	git fetch git://github.com/jrl-umi3218/jrl-travis.git master
	git subtree pull --prefix .travis git://github.com/jrl-umi3218/jrl-travis.git master --squash
