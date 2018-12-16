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


    # types
    from .Plan import Plan as newPlan


    # protocol obligations
    @ampcor.provides
    def estimate(self, plexus, reference, target, **kwds):
        """
        Estimate the offset field between a pair of raster images
        """
        # all done
        return 0


    # interface
    def makePlan(self, reference, target):
        """
        Formulate a computational plan for correlating {reference} and {target} to produce an
        offset map
        """
        # make a plan
        plan = self.newPlan()
        # and return it
        return plan


    def show(self, channel):
        """
        Display my configuration and details about the correlation plan
        """
        # show who i am
        channel.line(f" -- estimator: {self.pyre_family()}")
        # describe my coarse map strategy
        self.coarse.show(channel=channel)

        # all done
        return self


# end of file
