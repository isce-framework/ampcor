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


// put everything in my private namespace
namespace ampcor {
    namespace extension {
        namespace cuda {

            // the module method table
            PyMethodDef module_methods[] = {
                 // the copyright method
                 { copyright__name__, copyright, METH_VARARGS, copyright__doc__ },
                 // the version
                 { version__name__, version, METH_VARARGS, version__doc__ },

                 // image registration
                 { sequential::alloc__name__,
                   sequential::alloc, METH_VARARGS, sequential::alloc__doc__},

                 { sequential::addReference__name__,
                   sequential::addReference, METH_VARARGS, sequential::addReference__doc__},

                 { sequential::addTarget__name__,
                   sequential::addTarget, METH_VARARGS, sequential::addTarget__doc__},

                 { sequential::adjust__name__,
                   sequential::adjust, METH_VARARGS, sequential::adjust__doc__},

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
                "ampcor_cuda",
                // the module documentation string
                __doc__,
                // size of the per-interpreter state of the module; -1 if this state is global
                -1,
                // the methods defined in this module
                module_methods
            };

        } // of namespace cuda
    } // of namespace extension
} // of namespace ampcor


// initialization function for the module
// *must* be called PyInit_ampcor_cuda
PyMODINIT_FUNC
PyInit_ampcor_cuda()
{
    // create the module
    PyObject * module = PyModule_Create(&ampcor::extension::cuda::module_definition);
    // check whether module creation succeeded and raise an exception if not
    if (!module) {
        return module;
    }

    // otherwise, we have an initialized module; register the package exceptions
    ampcor::extension::cuda::registerExceptionHierarchy(module);
    // and return the newly created module
    return module;
}

// end of file
