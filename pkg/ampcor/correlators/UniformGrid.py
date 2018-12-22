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
        # get my shape
        shape = self.shape
        # split {bounds} into evenly spaced swaths
        swaths = tuple(b//s for b,s in zip(bounds, shape))
        # compute the unallocated border around the raster
        margin = tuple(b%s for b,s in zip(bounds, shape))




        print(f"ampcor.correlators.UniformGrid.points:")
        print(f"   grid={shape}")
        print(f"   bounds={bounds}")
        print(f"   swaths={swaths}")
        print(f"   margin={margin}")

        raise SystemExit(0)
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
