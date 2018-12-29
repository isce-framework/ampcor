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
        # make a channel
        channel = plexus.info
        # grab my timer
        timer = self.timer

        # show me
        channel.log(f"correlator: {self.pyre_family()}")

        # choose the correlator implementation
        worker = self.makeWorker(layout=plexus.shell)
        # show me
        channel.log(f"worker: {worker}")

        # start the timer
        timer.reset().start()
        # make a plan
        plan = self.makePlan(reference, target)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"correlation plan: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # open the two rasters and get access to the data
        ref = reference.open().raster
        tgt = target.open().raster
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"opened the two rasters: {1e3 * timer.read():.3f} ms")

        # restart the timer
        timer.reset().start()
        # make a plan
        offsets = worker.correlate(reference=ref, target=tgt, plan=plan, channel=channel)
        # stop the timer
        timer.stop()
        # show me
        channel.log(f"offset estimation: {1e3 * timer.read():.3f} ms")

        # all done
        return 0


    # interface
    def makeWorker(self, layout):
        """
        Deduce the correlator implementation strategy
        """
        # if the user asked for GPU acceleration and we support it
        if layout.gpus and ampcor.ext.libampcor_cuda:
            # use the GPU aware implementation
            from .CUDA import CUDA as workerFactory
        # if the CPU acceleration is available
        elif ampcor.ext.libampcor:
            # use the native implementation
            from .Sequential import Sequential as workerFactory
        # otherwise
        else:
            # complain
            raise NotImplementedError("no available correlation strategy")

        # instantiate
        worker = workerFactory()
        # that's all until there is support for other types of parallelism
        return worker


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


    # private data
    timer = ampcor.executive.newTimer(name="ampcor.mga")


# end of file
