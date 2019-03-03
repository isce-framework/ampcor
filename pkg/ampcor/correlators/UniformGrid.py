# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


# externals
import itertools
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
        # split {bounds} into evenly spaced tiles
        tile = tuple(b//s for b,s in zip(bounds, shape))
        # compute the unallocated border around the raster
        margin = tuple(b%s for b,s in zip(bounds, shape))
        # build the sequences of coordinates for tile centers along each axis
        ticks = tuple(
            # by generating the locations
            tuple(m//2 + n*t + t//2 for n in range(g))
            # given the layout of each axis
            for g, m, t in zip(shape, margin, tile)
        )
        # their cartesian product generates the centers of all the tiles in the grid
        centers = tuple(itertools.product(*ticks))
        # all done
        return centers


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
