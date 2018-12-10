// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//


#if !defined(ampcor_extension_cuda_capsules_h)
#define ampcor_extension_cuda_capsules_h


// capsules
namespace ampcor {
    namespace extension {
        namespace cuda {

            // sequential
            namespace sequential {
                void free(PyObject *);
                const char * const capsule_t = "ampcor::cuda::sequential::capsule_t";
            } // of namespace slc

        } // of namespace cuda
    } // of namespace extension
} // of namespace ampcor


#endif

// end of file
