# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2018 all rights reserved
#


# framework
import ampcor
# protocols
from .Raster import Raster as raster


# data product factories
def newSLC():
    """
    Create an SLC raster object in its default configuration
    """
    # get the component
    from .SLC import SLC
    # instantiate and return it
    return SLC()


# data product foundries; these get used by the framework during component binding
@ampcor.foundry(implements=raster, tip="an SLC raster image")
def slc():
    # get the component
    from .SLC import SLC
    # borrow its docstring
    __doc__ = SLC.__doc__
    # and publish it
    return SLC


# end of file
