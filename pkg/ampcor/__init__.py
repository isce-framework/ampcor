# -*- Python -*-
# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis <michael.aivazis@para-sim.com>
# parasim
# (c) 1998-2019 all rights reserved
#


# import and publish pyre symbols
from pyre import (
    # protocols, components, traits, and their infrastructure
    schemata, constraints, properties, protocol, component, foundry,
    # decorators
    export, provides,
    # the manager of the pyre runtime
    executive,
    # support for concurrency
    nexus,
    # support for multidimensional containers
    grid,
    # miscellaneous
    tracking, units
    )


# bootstrap
package = executive.registerPackage(name='ampcor', file=__file__)
# save the geography
home, prefix, defaults = package.layout()

# publish local modules
from . import (
    meta,         # package meta-data
    exceptions,   # the exception hierarchy
    dom,          # the data model
    correlators,  # the image correlators
    cli,          # the command line interface
    shells,       # the supported application shells
)


# administrivia
def copyright():
    """
    Return the copyright note
    """
    # pull and print the meta-data
    return print(meta.header)


def license():
    """
    Print the license
    """
    # pull and print the meta-data
    return print(meta.license)


def built():
    """
    Return the build timestamp
    """
    # pull and return the meta-data
    return meta.date


def credits():
    """
    Print the acknowledgments
    """
    return print(meta.acknowledgments)


def version():
    """
    Return the version
    """
    # pull and return the meta-data
    return meta.version


# end of file
