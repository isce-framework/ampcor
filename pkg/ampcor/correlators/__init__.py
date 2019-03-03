# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


# pull in the command decorators
from .. import foundry

# protocols
from .Correlator import Correlator as correlator
from .Domain import Domain as domain
from .Functor import Functor as functor
from .Offsets import Offsets as offsets


# correlation strategies
@foundry(implements=correlator, tip="estimate an offset field using MGA's implementation")
def mga():
    # get the action
    from .MGA import MGA
    # borrow its doctsring
    __doc__ = MGA.__doc__
    # and publish it
    return MGA


# strategies for placing tiles on the reference and target rasters
@foundry(implements=offsets, tip="a grid based generators of a coarse offset map")
def grid():
    # get the action
    from .Grid import Grid
    # borrow its doctsring
    __doc__ = Grid.__doc__
    # and publish it
    return Grid


# strategies for laying tiles on the reference raster
@foundry(implements=domain, tip="generate a uniform grid of reference points")
def uniform():
    # get the action
    from .UniformGrid import UniformGrid
    # borrow its doctsring
    __doc__ = UniformGrid.__doc__
    # and publish it
    return UniformGrid


# generators of points on the target raster
@foundry(implements=offsets, tip="a functor that applies a constant shift")
def constant():
    # get the action
    from .Constant import Constant
    # borrow its doctsring
    __doc__ = Constant.__doc__
    # and publish it
    return Constant


# end of file
