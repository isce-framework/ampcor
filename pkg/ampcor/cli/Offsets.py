# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
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

    correlator = ampcor.correlators.correlator()
    correlator.doc = "the calculator of the offset field"


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
        # grab the correlator
        correlator = self.correlator
        # and ask it to do its thing
        return correlator.estimate(plexus=plexus, reference=reference, target=target, **kwds)


    @ampcor.export(tip="display my configuration")
    def info(self, plexus, **kwds):
        """
        Display the action configuration
        """
        # grab a channel
        channel = plexus.info

        # get things going
        channel.line()

        # shell
        channel.line(f" -- shell: {plexus.shell}")
        channel.line(f"    hosts: {plexus.shell.hosts}")
        channel.line(f"    tasks: {plexus.shell.tasks} per host")
        channel.line(f"    gpus:  {plexus.shell.gpus} per task")

        # inputs
        channel.line(f" -- data files")
        # reference raster
        channel.line(f"    reference: {self.reference}")
        if self.reference:
            channel.line(f"        data: {self.reference.data}")
            channel.line(f"        shape: {self.reference.shape}")
            channel.line(f"        size: {self.reference.size()} bytes")
        # target raster
        channel.line(f"    target: {self.target}")
        if self.target:
            channel.line(f"        data: {self.target.data}")
            channel.line(f"        shape: {self.target.shape}")
            channel.line(f"        size: {self.target.size()} bytes")

        # show the correlator configuration
        self.correlator.show(channel=channel)
        # ask it to make a plan
        plan = self.correlator.makePlan(reference=self.reference, target=self.target)
        # and show the plan details
        plan.show(channel=channel)

        # flush
        channel.log()

        # all done; indicate success
        return 0


# end of file
