# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor


# declaration
class CUDA:
    """
    The CUDA accelerated correlation strategy
    """


    # interface
    def refine(self, rasters, plan, channel):
        """
        Correlate a pair rasters given a plan
        """


# end of file