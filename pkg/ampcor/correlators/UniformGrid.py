# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor
# my protocol
from .Scanner import Scanner as scanner


# declaration
class UniformGrid(ampcor.component,
                  family="ampcor.correlators.scanners.uniform", implements=scanner):
    """
    A scanner that places reference tiles on a uniform grid
    """


    # user configurable state
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (1,1)
    shape.doc = "the shape of the grid of points to generate"


    # protocol requirements
    @empcor.export
    def scan(self, raster, **kwds):
        """
        Generate a uniform grid of points over {raster}
        """
        # all done
        return


# end of file
