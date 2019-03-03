# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# get the framework
import pyre

# the base class for my exceptions
class AmpcorError(pyre.PyreError):
    """
    Base class for all ampcor errors
    """

# component configuration errors
class ConfigurationError(AmpcorError):
    """
    Exception raised when ampcor components detect inconsistencies in their configurations
    """

    # public data
    description = "configuration error: {0.reason}"

    # meta-methods
    def __init__(self, reason, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.reason = reason
        # all done
        return


# end of file
