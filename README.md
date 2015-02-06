[![Build Status](https://travis-ci.org/jorisv/Eigen3ToPython.svg?branch=master)](https://travis-ci.org/jorisv/Eigen3ToPython)

Eigen3ToPython aim to make a bidirectional bridge between numpy and eigen3.

This python library goal is not to provide a full eigen3 python binding but to provide easy conversion between numpy and eigen3.

Pulling git subtree
------

To update sync cmake or .travis directory with their upstream git repository:

	git fetch git://github.com/jrl-umi3218/jrl-cmakemodules.git master
	git subtree pull --prefix cmake git://github.com/jrl-umi3218/jrl-cmakemodules.git master --squash

	git fetch git://github.com/jrl-umi3218/jrl-travis.git master
	git subtree pull --prefix .travis git://github.com/jrl-umi3218/jrl-travis.git master --squash
