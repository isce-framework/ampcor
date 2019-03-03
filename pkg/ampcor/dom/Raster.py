# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


# framework
import ampcor


# declaration
class Raster(ampcor.protocol, family="ampcor.dom.rasters"):
    """
    The base class for all pixel based data products
    """


    # public data
    shape = ampcor.properties.tuple(schema=ampcor.properties.int())
    shape.doc = "the shape of the raster in pixels"

    data = ampcor.properties.path()
    data.doc = "the path to my binary data"


    # requirements
    @ampcor.provides
    def size(self):
        """
        Compute my memory footprint
        """

    @ampcor.provides
    def slice(self, begin, end):
        """
        Grant access to a slice of my data bound by the index pair {begin} and {end}
        """

    @ampcor.provides
    def open(self, filename, mode="r"):
        """
        Map me over the contents of {filename}
        """


# end of file
