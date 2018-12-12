# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# pull the action protocol
from .Action import Action as action
# and the base panel
from .Command import Command as command
# pull in the command decorator
from .. import foundry


# commands
@foundry(implements=action, tip="estimate an offset field given a pair of raster images")
def offsets():
    # get the action
    from .Offsets import Offsets
    # borrow its doctsring
    __doc__ = Offsets.__doc__
    # and publish it
    return Offsets


# help
@foundry(implements=action, tip="display information about this application")
def about():
    # get the action
    from .About import About
    # borrow its docstring
    __doc__ = About.__doc__
    # and publish it
    return About


# end of file
