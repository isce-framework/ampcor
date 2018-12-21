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
from .Domain import Domain as domain


# declaration
class UniformGrid(ampcor.component,
                  family="ampcor.correlators.domains.uniform", implements=domain):
    """
    A domain that generates domain points on a uniform grid
    """


    # user configurable state
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (1,1)
    shape.doc = "the shape of the grid of points to generate"


    # protocol requirements
    @ampcor.export
    def points(self, bounds, **kwds):
        """
        Generate a cloud of points within {extent} where reference tiles will be placed
        """
        print(f"ampcor.correlators.UniformGrid.points: bounds={bounds}")
        # all done
        return


    # interface
    def show(self, channel):
        """
        Display my configuration
        """
        # show who i am
        channel.line(f" -- domain: {self.pyre_family()}")
        channel.line(f"        shape: {self.shape}")
        # all done
        return


# end of file
