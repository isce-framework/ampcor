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
class Domain(ampcor.protocol, family="ampcor.correlators.domains"):
    """
    The protocol that must be satisfied by all components that generate collections of
    reference tiles
    """


    # requirements
    @ampcor.provides
    def points(self, bounds, **kwds):
        """
        Generate a cloud of points within {bounds} where reference tiles will be placed
        """


    # hooks
    @classmethod
    def pyre_default(self, **kwds):
        """
        Provide a default implementation
        """
        # pull the uniform grid generator
        from .UniformGrid import UniformGrid
        # and publish it
        return UniformGrid


# end of file
