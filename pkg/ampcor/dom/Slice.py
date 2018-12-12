# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor


# declaration
class Slice:
    """
    Encapsulation of a portion of a raster
    """


    # public data
    raster = None
    begin = None
    end = None


    # meta-methods
    def __init__(self, raster, begin, end, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the raster
        self.raster = raster
        # and the indices
        self.begin = begin
        self.end = end

        # all done
        return


    def __getitem__(self, index):
        """
        Fetch the item in {raster} at {index}
        """
        # adjust the index relative to the {raster}
        index = tuple(b+i for b,i in zip(self.begin, index))
        # fetch the value
        return self.raster[index]


# end of file
