// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//

#if !defined(ampcor_extension_cuda_exceptions_h)
#define ampcor_extension_cuda_exceptions_h


// place everything in my private namespace
namespace ampcor {
    namespace extension {
        namespace cuda {

            // exception registration
            PyObject * registerExceptionHierarchy(PyObject *);

        } // of namespace cuda
    } // of namespace extension
} // of namespace ampcor

#endif

// end of file
