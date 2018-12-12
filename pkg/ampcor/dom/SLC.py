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
    def slice(self, begin, end):
        """
        Grant access to a slice of my data bound by the index pair {begin} and {end}
        """
        # return the slice
        return self.sliceFactory(raster=self, begin=begin, end=end)


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
    def __getitem__(self, index):
        """
        Fetch data at the given index
        """
        # attempt to
        try:
            # to convert the {index} into an int
            index = int(index)
        # If this fails
        except TypeError:
            # hand it to the binding that retrieves data using tuples as indices
            return libampcor.slc_atIndex(self.slc, *index)

        # if successful, the {index} is an offset
        return libampcor.slc_atOffset(self.slc, int(index))


    # implementation details
    # private data
    slc = None


# end of file
