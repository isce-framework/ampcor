# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


# framework
import ampcor


# declaration
class Correlator(ampcor.protocol, family="ampcor.correlators"):
    """
    The protocol for all AMPCOR correlator implementations
    """


    # requirements
    @ampcor.provides
    def estimate(self, plexus, reference, target):
        """
        Estimate the offset field between a pair of raster images
        """


    # hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # pull mga's implementation
        from .MGA import MGA
        # and publish it
        return MGA


# end of file
