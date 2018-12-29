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
class Sequential:
    """
    The sequential correlation strategy
    """


    # interface
    def correlate(self, reference, target, plan, channel):
        """
        Correlate two rasters given a plan
        """
        # get the bindings
        libampcor = ampcor.ext.libampcor
        # instantiate the worker
        worker = libampcor.sequential()

        # all done
        return


# end of file
