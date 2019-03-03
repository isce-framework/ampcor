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
from .Offsets import Offsets


# declaration
class Grid(ampcor.component, family="ampcor.correlators.offsets.grid", implements=Offsets):
    """
    The protocol for initial guesses for the offset map
    """


    # user configurable state
    domain = ampcor.correlators.domain()
    domain.doc = "the domain of the map"

    functor = ampcor.correlators.functor()
    functor.doc = "the function that maps points from the reference raster to the target raster"


    # requirements
    @ampcor.export
    def map(self, reference, **kwds):
        """
        Build an offset map between {reference} and {target}
        """
        # get my domain
        domain = self.domain
        # and the functor that generates the codomain
        functor = self.functor
        # make a map
        offmap = ampcor.dom.newOffsetMap(shape=domain.shape)
        # generate the reference points and attach them as the domain of the offset map
        offmap.domain = tuple(domain.points(bounds=reference.shape))
        # invoke the map to generate the corresponding points on the target image
        offmap.codomain = tuple(functor.codomain(domain=offmap.domain))
        # all done
        return offmap


    # interface
    def show(self, channel):
        """
        Display my configuration
        """
        # show who i am
        channel.line(f" -- offsets: {self.pyre_family()}")
        # show my domain
        self.domain.show(channel=channel)
        # and my codomain generator
        self.functor.show(channel=channel)
        # all done
        return


# end of file
