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
from .Offsets import Offsets


# declaration
class Grid(ampcor.component, family="ampcor.correlators.offsets.grid", implements=Offsets):
    """
    The protocol for initial guesses for the offset map
    """


    # user configurable state
    domain = ampcor.correlators.domain()
    domain.doc = "the domain of the map"




    # requirements
    @ampcor.export
    def map(self, **kwds):
        """
        Build an offset map between {reference} and {target}
        """


    # interface
    def show(self, channel):
        """
        Display my configuration
        """
        # show who i am
        channel.line(f" -- offsets: {self.pyre_family()}")
        # show my domain
        self.domain.show(channel=channel)
        # all done
        return


# end of file
