# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# the framework
import ampcor
# the extension
from ampcor.ext import ampcor as libampcor
# my protocol
from .Raster import Raster


# declaration
class SLC(ampcor.component, family="ampcor.dom.rasters.slc", implements=Raster):
    """
    Access to the data of a file based SLC
    """


    # types
    from .Slice import Slice as sliceFactory


    # constants
    # the memory footprint of individual pixels
    pixelSize = libampcor.slc_pixelSize()


    # user configurable state
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.default = (1,1)
    shape.doc = "the shape of the raster: lines x samples"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # protocol obligations
    @ampcor.export
    def size(self):
        """
        Compute my memory footprint
        """
        # unpack
        lines, samples = self.shape
        # compute and return
        return lines * samples * self.pixelSize


    @ampcor.export
    def slice(self, begin, shape):
        """
        Grant access to a slice of my data bound by the index pair {begin} and {end}
        """
        # N.B.: {end} follows the one-past-the-end of the range convention, just like {range}
        # this means that overflow doesn't happen unless {end} > {self.shape}

        # go through the index that describes the beginning of the slice
        for b in begin:
            # if any of them are negative
            if b < 0:
                # indicate that this is an invalid slice
                return None
        # make sure the indices in {end} don't overflow
        for b, s, l in zip(begin, shape, self.shape):
            # if any of them do
            if b+s > s:
                # indicate this is an invalid slice
                return None

        # if all goes well, make a slice and return it
        return self.sliceFactory(raster=self, begin=begin, shape=shape)


    @ampcor.export
    def open(self, mode="r"):
        """
        Map me over the contents of {filename}
        """
        # unpack my shape
        lines, samples = self.shape
        # build an SLC raster image over the contents of {filename}
        self.slc = libampcor.slc_map(filename=self.data.path, lines=lines, samples=samples)
        # all done
        return self


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # make a tile out of my shape
        self.tile = ampcor.grid.tile(shape=self.shape)
        # all done
        return


    def __getitem__(self, index):
        """
        Fetch data at the given index
        """
        # convert {index} into an offset
        offset = self.tile.offset(index)
        # grab the data and return it
        return libampcor.slc_getitem(self.slc, index)


    # implementation details
    # private data
    slc = None


# end of file
