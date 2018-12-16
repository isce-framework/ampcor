# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor


# declaration
class OffsetMap:
    """
    A logically Cartesian map that establishes a correspondence between a collection of points
    on a {reference} raster and a {target} raster
    """


    # public data
    @property
    def size(self):
        """
        Compute the total number of elements in the map
        """
        # easy enough
        return self.tile.size


    @property
    def layout(self):
        """
        Return the index packing order
        """
        # easy enough
        self.tile.layout


    # meta-methods
    def __init__(self, shape, laoyout=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # storage
        self.domain = []
        self.codomain = []
        # access as a Cartesian map
        self.tile = ampcor.grid.tile(shape=shape, layout=layout)
        # all done
        return


    def __getitem__(self, index):
        """
        Return the pair of correlated points stored at {index}
        """
        # attempt to
        try:
            # cast {index} to an integer
            index = int(index)
        # if this fails
        except TypeError:
            # ask my tile for the offset
            index = self.tile.offset(index)

        # in any case, pull the corresponding points
        ref = self.domain[index]
        tgt = self.codomain[index]
        # and return them
        return (ref, tgt)


    def __setitem__(self, index, points):
        """
        Return the value stored at {index}
        """
        # attempt to
        try:
            # cast {index} to an integer
            index = int(index)
        # if this fails
        except TypeError:
            # let my tile do the work
            index = self.tile.offset(index)

        # unpack the points
        ref, tgt = points
        # store them
        self.domain[index] = ref
        self.codomain[index] = tgt

        # all done
        return


    def __len__(self):
        """
        Compute my length
        """
        # delegate to the corresponding property
        return self.size


# end of file
