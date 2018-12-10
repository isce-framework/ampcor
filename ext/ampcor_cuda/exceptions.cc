// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//

#include <portinfo>
#include <Python.h>
#include <string>

#include "exceptions.h"

namespace ampcor {
    namespace extension {
        namespace cuda {
            // base class for ampcor errors
            std::string Error__name__ = "Error";
            PyObject * Error = 0;
        } // of namespace cuda
    } // of namespace extension
} // of namespace ampcor


// exception registration
PyObject *
ampcor::extension::cuda::
registerExceptionHierarchy(PyObject * module) {

    std::string stem = "ampcor.";

    // the base class
    // build its name
    std::string errorName = stem + Error__name__;
    // and the exception object
    Error = PyErr_NewException(errorName.c_str(), 0, 0);
    // increment its reference count so we can pass ownership to the module
    Py_INCREF(Error);
    // register it with the module
    PyModule_AddObject(module, Error__name__.c_str(), Error);

    // and return the module
    return module;
}

// end of file
