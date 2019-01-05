#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


"""
Sanity check: verify that the extension from the {ampcor} package is accessible
"""


def test():
    # access the {ampcor} extension
    from ampcor.ext import libampcor
    # all done
    return 0


# main
if __name__ == "__main__":
    # do...
    status = test()
    # share
    raise SystemExit(status)


# end of file
