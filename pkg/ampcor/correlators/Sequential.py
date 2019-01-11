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
class Sequential:
    """
    The sequential registration strategy
    """


    # interface
    def adjust(self, rasters, plan, channel):
        """
        Correlate two rasters given a plan
        """
        # unpack the rasters
        ref, tgt = rasters
        # ask the plan for the total number of points on the map
        points = len(plan)
        # the shape of the reference chips
        chip = plan.chip
        # and the shape of the search windows
        window = plan.window

        # get the bindings
        libampcor = ampcor.ext.libampcor
        # instantiate the worker
        worker = libampcor.sequential(points, chip, window)

        # go through the valid pairs
        for idx, (r,t) in enumerate(plan.pairs):
            # load the reference slice
            libampcor.addReference(worker, ref, idx, r.begin, r.end)
            # load the target slice
            libampcor.addTarget(worker, tgt, idx, t.begin, t.end)

        # ask the worker to perform pixel level adjustments
        libampcor.adjust(worker)

        # all done
        return


# end of file
