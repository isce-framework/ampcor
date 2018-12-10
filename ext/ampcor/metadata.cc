// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//

// configuration
#include <portinfo>

// externals
#include <Python.h>
#include <ampcor/meta/version.h>

// my declarations
#include "metadata.h"


// copyright
const char * const
ampcor::extension::
copyright__name__ = "copyright";

const char * const
ampcor::extension::
copyright__doc__ = "the project copyright string";

PyObject *
ampcor::extension::
copyright(PyObject *, PyObject *)
{
    // build the note
    const char * const copyright_note =
        "ampcor: (c) 1998-2018 michael a.g. aïvázis <michael.aivazis@para-sim.com>";
    // and return it
    return Py_BuildValue("s", copyright_note);
}


// version
const char * const
ampcor::extension::
version__name__ = "version";

const char * const
ampcor::extension::
version__doc__ = "the project version string";

PyObject *
ampcor::extension::
version(PyObject *, PyObject *)
{
    return Py_BuildValue("s", ampcor::meta::version());
}


// end of file
