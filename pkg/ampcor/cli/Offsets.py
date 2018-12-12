# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# externals
import ampcor


# declaration
class Offsets(ampcor.cli.command, family="ampcor.cli.offsets"):
    """
    Estimate an offset field given a pair of rasters
    """


    # user configurable state
    reference = ampcor.dom.raster()
    reference.doc = "the reference raster image"

    target = ampcor.dom.raster()
    target.doc = "the target raster image"


    # behaviors
    @ampcor.export(tip="produce the offset field between the {reference} and {target} rasters")
    def estimate(self, plexus, **kwds):
        """
        Produce an offset map between the {reference} and {target} images
        """
        # get the reference image and open it
        reference = self.reference.open()
        # repeat for the target image
        target = self.target.open()

        # all done
        return 0


# end of file
