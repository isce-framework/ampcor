# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# externals
import functools
import operator
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

    shape = None
    size = None
    footprint = None


    # interface
    def show(self, channel):
        """
        Display my vitals in {channel}
        """
        # sign on
        channel.line(f"                begin: {self.begin}")
        channel.line(f"                shape: {self.shape}")
        channel.line(f"                end: {self.end}")
        channel.line(f"                size: {self.size} cells")
        channel.line(f"                footprint: {self.footprint} bytes")
        # all done
        return


    # meta-methods
    def __init__(self, raster, begin, shape, **kwds):
        # chain up
        super().__init__(**kwds)

        # save the raster
        self.raster = raster
        # the start of the slice
        self.begin = begin
        # the end of the slice
        self.end = tuple(b+s for b,s in zip(begin, shape))
        # its shape
        self.shape = shape
        # size
        self.size = functools.reduce(operator.mul, shape, 1)
        # and footprint
        self.footprint = self.size * raster.pixelSize

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
