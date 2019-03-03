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
class Functor(ampcor.protocol, family="ampcor.correlators.functors"):
    """
    The protocol implemented by generators of points for the target raster
    """


    # requirements
    @ampcor.provides
    def codomain(self, domain, **kwds):
        """
        Given points on the {reference} raster in {domain}, generate their images in {target}
        """


    # framework hooks
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Provide a default implementation
        """
        # pull the default implementation
        from .Constant import Constant
        # and publish it
        return Constant


# end of file
