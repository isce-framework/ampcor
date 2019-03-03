#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


"""
Sanity check: verify that the {ampcor} package is accessible
"""


def test():
    # access the {ampcor} package
    import ampcor
    # and the journal
    import journal

    # activate some channels
    # journal.debug("pyre.memory.direct").activate()

    # make one
    channel = journal.debug("ampcor.slc")#.activate()

    # create an SLC
    slc = ampcor.dom.newSLC()
    # configure it
    slc.shape = 36864, 10344
    slc.data = "../../data/20061231.slc"

    # show me its size
    channel.log(f"slc: {slc.size()/1024**3:.3f} Gb")
    # load some real data
    slc.open()

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
