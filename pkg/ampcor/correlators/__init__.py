# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# pull in the command decorators
from .. import foundry

# protocols
from .Scanner import Scanner as scanner


# scanners
@foundry(implements=scanner, tip="generate a uniform grid of reference points")
def uniform():
    # get the action
    from .UniformGrid import UniformGrid
    # borrow its doctsring
    __doc__ = UniformGrid.__doc__
    # and publish it
    return UniformGrid


# end of file
