#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


"""
Sanity check: verify that the {ampcor} package is accessible
"""


def test():
    # access the {ampcor} package
    import ampcor
    # and the journal
    import journal

    # make a channel
    channel = journal.debug("ampcor.slc")

    # activate some channels
    # journal.debug("pyre.memory.direct").activate()
    # journal.debug("ampcor.slc").activate()

    # create an SLC
    slc = ampcor.dom.newSLC()
    # configure it
    slc.shape = 36864, 10344
    slc.data = "../../data/20061231.slc"

    # load the data
    slc.open()
    # verify that the two access modes produce the same result
    assert slc[0] == slc[0,0]
    # show me
    channel.log(f"slc[0,0] = {slc[0,0]}")

    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
