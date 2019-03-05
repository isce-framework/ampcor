// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_kernels_h)
#define ampcor_libampcor_cuda_correlators_kernels_h


// forward declarations
namespace ampcor {
    namespace cuda {
        namespace kernels {

            // compute amplitudes of the tile pixels
            void detect(std::complex<float> * cArena, std::size_t cells, float * rArena);

        }
    }
}

// code guard
#endif

// end of file
