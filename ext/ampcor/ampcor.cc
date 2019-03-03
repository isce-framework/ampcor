// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

#include <portinfo>
#include <Python.h>

// the module method declarations
#include "exceptions.h"
#include "metadata.h"
#include "sequential.h"
#include "slc.h"


// put everything in my private namespace
namespace ampcor {
    namespace extension {

        // the module method table
        PyMethodDef module_methods[] = {
            // the copyright method
            { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
            // the version
            { version__name__, version, METH_VARARGS, version__doc__ },

            // correlation plan
            { sequential::alloc__name__,
              sequential::alloc, METH_VARARGS, sequential::alloc__doc__},

            { sequential::addReference__name__,
              sequential::addReference, METH_VARARGS, sequential::addReference__doc__},

            { sequential::addTarget__name__,
              sequential::addTarget, METH_VARARGS, sequential::addTarget__doc__},

            { sequential::adjust__name__,
              sequential::adjust, METH_VARARGS, sequential::adjust__doc__},

            { sequential::refine__name__,
              sequential::refine, METH_VARARGS, sequential::refine__doc__},

            // slc support
            // pixel size
            { slc::pixelSize__name__, slc::pixelSize, METH_NOARGS, slc::pixelSize__doc__},
            // memory map over an existing file
            { slc::map__name__,
              reinterpret_cast<PyCFunction>(slc::map), METH_VARARGS | METH_KEYWORDS,
              slc::map__doc__},
            // fetch data at a given offset
            { slc::getitem__name__, slc::getitem, METH_VARARGS, slc::getitem__doc__},

            // sentinel
            { 0, 0, 0, 0 }
        };

        // the module documentation string
        const char * const __doc__ = "ampcor: a tool for benchmarking ampcor implementations";

        // the module definition structure
        PyModuleDef module_definition = {
            // header
            PyModuleDef_HEAD_INIT,
            // the name of the module
            "ampcor",
            // the module documentation string
            __doc__,
            // size of the per-interpreter state of the module; -1 if this state is global
            -1,
            // the methods defined in this module
            module_methods
        };

    } // of namespace extension
} // of namespace ampcor


// initialization function for the module
// *must* be called PyInit_ampcor
PyMODINIT_FUNC
PyInit_ampcor()
{
    // create the module
    PyObject * module = PyModule_Create(&ampcor::extension::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }

    // otherwise, we have an initialized module; register the package exceptions
    ampcor::extension::registerExceptionHierarchy(module);
    // and return the newly created module
    return module;
}

// end of file
