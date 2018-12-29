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
    def correlate(self, reference, target, plan, channel):
        """
        Correlate two rasters given a plan
        """
# end of file
