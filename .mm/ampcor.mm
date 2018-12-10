# -*- Makefile -*-
#
# michael a.g. aïvázis
# parasim
# (c) 1998-2018 all rights reserved
#


# project meta-data
ampcor.major := 1
ampcor.minor := 0

# ampcor consists of a python package
ampcor.packages := ampcor.pkg
# libraries
ampcor.libraries :=
# python extensions
ampcor.extensions :=
# and some tests
ampcor.tests := ampcor.pkg.tests

# the ampcor package meta-data
ampcor.pkg.stem := ampcor
ampcor.pkg.drivers := ampcor

# the ampcor test suite
ampcor.pkg.tests.stem := ampcor
ampcor.pkg.tests.prerequisites := ampcor.pkg ampcor.ext

# base functionality
ampcor.assets := libampcor
# optional CUDA acceleration
ampcor.assets += ${if ${value cuda.dir},libampcor_cuda,}

# get the asset definitions
include ${addsuffix .def,$(ampcor.assets)}

# end of file
