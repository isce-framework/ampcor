// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//


#if !defined(ampcor_extension_capsules_h)
#define ampcor_extension_capsules_h


// capsules
namespace ampcor {
    namespace extension {

        // sequential worker
        namespace sequential {
            void free(PyObject *);
            const char * const capsule_t = "ampcor::sequential::capsule_t";
        } // of namespace slc

        // slc
        namespace slc {
            void free(PyObject *);
            const char * const capsule_t = "ampcor::slc::capsule_t";
        } // of namespace slc

    } // of namespace extension
} // of namespace ampcor


#endif

// end of file
