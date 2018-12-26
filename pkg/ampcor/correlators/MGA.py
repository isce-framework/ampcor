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
from .Correlator import Correlator


# declaration
class MGA(ampcor.component, family="ampcor.correlators.mga", implements=Correlator):
    """
    MGA's implementation of the offset field estimator
    """


    # user configurable state
    coarse = ampcor.correlators.offsets()
    coarse.doc = "the initial guess for the offset map"

    # control over the correlation plan
    chip = ampcor.properties.tuple(schema=ampcor.properties.int())
    chip.default = 128, 128
    chip.doc = "the shape of the reference chip"

    padding = ampcor.properties.tuple(schema=ampcor.properties.int())
    padding.default = 32, 32
    padding.doc = "padding around the chip shape to form the search window in the target raster"


    # types
    from .Plan import Plan as newPlan


    # protocol obligations
    @ampcor.provides
    def estimate(self, plexus, reference, target, **kwds):
        """
        Estimate the offset field between a pair of raster images
        """
        # make a plan
        plan = self.makePlan(reference, target)
        # make a channel
        channel = plexus.info

        # show me
        channel.log(f"correlator: {self.pyre_family()}")
        channel.log(f"plan: {plan}")

        # all done
        return 0


    # interface
    def makePlan(self, reference, target):
        """
        Formulate a computational plan for correlating {reference} and {target} to produce an
        offset map
        """
        # form the coarse map
        coarse = self.coarse.map(reference=reference)
        # make a plan
        plan = self.newPlan(correlator=self, offsets=coarse, rasters=(reference, target))
        # and return it
        return plan


    def show(self, channel):
        """
        Display my configuration and details about the correlation plan
        """
        # show who i am
        channel.line(f" -- estimator: {self.pyre_family()}")
        # display the reference chip size
        channel.line(f"        chip: {self.chip}")
        # and the search window padding
        channel.line(f"        padding: {self.padding}")
        # describe my coarse map strategy
        self.coarse.show(channel=channel)

        # all done
        return self


# end of file
