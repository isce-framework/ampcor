# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor


# declaration
class Plan:
    """
    Encapsulation of the computational work necessary to produce the offset map between a
    {reference} and a {target} image
    """

    # interface
    def assemble(self):
        """
        Put together the set of pairs of tiles to correlate
        """
        raise NotImplementedError("ampcor.correlators.Plan.assemble: NYI!")


    def show(self, channel):
        """
        Display details about this plan in {channel}
        """
        # sign on
        channel.line(f" -- plan:")
        # all done
        return


# end of file
