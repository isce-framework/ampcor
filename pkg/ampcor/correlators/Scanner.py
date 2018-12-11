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
class Scanner(ampcor.protocol, family="ampcor.correlators.scanners"):
    """
    The protocol that must be satisfied by all components that generate collections of
    reference tiles
    """


    # requirements
    @ampcor.provides
    def scan(self, raster, **kwds):
        """
        Generate a cloud of points over {raster} where reference tiles will be placed
        """

    # hooks
    def pyre_default(self, **kwds):
        """
        Provide a default implementation
        """
        # pull the uniform grid scanner
        from .UniformGrid import UniformGrid
        # and publish it
        return UniformGrid


# end of file
