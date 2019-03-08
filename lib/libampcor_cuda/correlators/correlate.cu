// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//


// configuration
#include <portinfo>
// STL
#include <complex>
// pyre
#include <pyre/journal.h>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pull the declarations
#include "public.h"


// the correlation kernel
template <typename value_t = float>
__global__
void
_correlate(const value_t * arena,
          const value_t * average,
          std::size_t rdim, std::size_t rcells,
          std::size_t tdim, std::size_t tcells,
          std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
          value_t * correlation);


// implementation
void
ampcor::cuda::kernels::
correlate(const float * dArena, const float * dAverage,
          std::size_t pairs,
          std::size_t refCells, std::size_t tgtCells, std::size_t corCells,
          std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
          float * dCorrelation)
{
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");

    // figure out the job layout and launch the calculation on the device
    // each thread block takes care of one tile pair, so we need as many blocks as there are pairs
    auto B = pairs;
    // the number of threads per block is determined by the shape of the reference  tile
    auto T = refDim;
    // each thread stores its partial sum in shared memory, so we need one {value_t}'s worth
    // of shared memory for each thread
    auto S = T * sizeof(float);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each, with "
        << S << " bytes of shared memory per block, for each of the " << corCells
        << " possible placements of the search window within the target tile;"
        << " a grand total of " << (B*corCells) << " kernel launches"
        << pyre::journal::endl;

    // for storing error codes
    cudaError_t status = cudaSuccess;
    // go through all possible row offsets for the sliding window
    for (auto rowOffset = 0; rowOffset < corDim; ++rowOffset) {
        // and all possible column offsets
        for (auto colOffset = 0; colOffset < corDim; ++colOffset) {
            // launch each kernel
            // N.B. kernel launch is an implicit barrier, so no need for any extra synchronization
            ::_correlate <<<B,T,S>>> (dArena, dAverage,
                                      refDim, refCells,
                                      tgtDim, tgtCells,
                                      corDim, rowOffset, colOffset,
                                      dCorrelation
                                      );
            // check for errors
            status = cudaPeekAtLastError();
            // if something went wrong
            if (status != cudaSuccess) {
                // make a channel
                pyre::journal::error_t channel("ampcor.cuda");
                // complain
                channel
                    << pyre::journal::at(__HERE__)
                    << "after launching the " << rowOffset << "x" << colOffset << " correlators: "
                    << cudaGetErrorName(status) << " (" << status << ")"
                    << pyre::journal::endl;
                // and bail
                break;
            }
        }
        // if something went wrong in the inner loop
        if (status != cudaSuccess) {
            // bail out of the outer loop as well
            break;
        }
    }
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // check
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while waiting for a kernel to finish: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // all done
    return;
}


// the correlation kernel
template <typename value_t>
__global__
void
_correlate(const value_t * arena, // the dataspace
           const value_t * average, // the hyper-grid of target amplitude averages
           std::size_t rdim, std::size_t rcells, // ref grid shape and size
           std::size_t tdim, std::size_t tcells, // tgt grid shape and size
           std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
           value_t * correlation) {

    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    // std::size_t T = blockDim.x;   // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    // std::size_t w = b*T + t;      // my worker id

    // get access to my shared memory
    extern __shared__ value_t scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // initialize my partial sum
    value_t num = 0;

    // reference and target grids area interleaved; compute the stride
    std::size_t stride = rcells + tcells;

    // my {ref} starting point is column {t} of grid {b}
    auto ref = arena + b*stride + t;
    // my {tgt} starting point is column {t} of grid {b} at (rowOffset, colOffset)
    // value_t * tgt = arena + b*stride + rcells + (rowOffset*tdim + colOffset) + t;
    // or, more simply
    auto tgt = ref + rcells + (rowOffset*tdim + colOffset);

    // run down the two columns
    for (std::size_t idx=0; idx < rdim; ++idx) {
        // fetch the ref value
        value_t r = ref[idx*rdim];
        // fetch the tgt value
        value_t t = tgt[idx*tdim];
        // update the numerator
        num += r * t;
    }

    // save my partial result
    scratch[t] = num;

    // barrier: make sure everybody is done updating shared memory with their partial sum
    cta.sync();

    // now, do the reduction in shared memory
    // N.B.: we assume the warp size is 32; this will need updates if the warp size changes
    if (t < 64) {
        // pull a neighbor's value
        num += scratch[t + 64];
        // and update my slot
        scratch[t] = num;
    }

    // barrier: make sure everybody is done updating shared memory with their partial sum
    cta.sync();

    // we are now within a warp
    if (t < 32) {
        // get a handle to the active thread group
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
        // pull the partial result from the second warp
        num += scratch[t + 32];
        // the power-of-2 threads
        for (int offset = 16; offset > 0; offset >>= 1) {
            // reduce using {shuffle}
            num += active.shfl_down(num, offset);
        }
    }

    // the master thread of each block
    if (t == 0) {
        // computes the slot where this result goes
        std::size_t slot = b*margin*margin + rowOffset*margin + colOffset;
        // and writes the sum to the result vector
        correlation[slot] = num;
    }

    // all done
    return;
}


// end of file
