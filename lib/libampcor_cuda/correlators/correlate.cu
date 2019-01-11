// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//


// configuration
#include <portinfo>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// pull the declarations
#include "public.h"


// local alias for the worker cell tupe
using cell_t = ampcor::cuda::correlators::CUDA::cell_type;

// the correlation kernel
__global__
static void
correlate(const cell_t * arena,
          const cell_t * average,
          std::size_t rdim, std::size_t rcells,
          std::size_t tdim, std::size_t tcells,
          std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
          cell_t * correlation);


// implementation
auto
ampcor::cuda::correlators::CUDA::
_newCorrelationMatrix(const cell_type * dArena, const cell_type * dAverage) const -> cell_type *
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::info_t tlog("ampcor.cuda.timings");

    // the total number of cells in the correlation matrix hyper-grid
    auto size = _pairs * _corCells;
    // with a memory footprint
    auto footprint = _pairs * _corFootprint;

    // pick a spot
    cell_type * dCorrelation = nullptr;
    // and make it point to a device buffer for the correlation matrix
    cudaError_t status = cudaMallocManaged(&dCorrelation, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the correlation matrix: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        return nullptr;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated " << footprint << " bytes for the correlation matrix at "
        << dCorrelation
        << pyre::journal::endl;

    // figure out the job layout and launch the calculation on the device
    // each thread block takes care of one tile pair, so we need as many blocks as there are pairs
    auto B = _pairs;
    // the number of threads per block is determined by the shape of the reference  tile
    auto T = _refShape[1];
    // each thread stores its partial sum in shared memory, so we need one {cell_type}'s worth
    // of shared memory for each thread
    auto S = T * sizeof(cell_type);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each, with "
        << S << " bytes of shared memory per block, for each of the " << _corCells
        << " possible placements of the search window within the target tile;"
        << " a grand total of " << (B*_corCells) << " kernel launches"
        << pyre::journal::endl;
    // start the clock
    timer.reset().start();
    // go through all possible row offsets for the sliding window
    for (auto rowOffset = 0; rowOffset < _corShape[0]; ++rowOffset) {
        // and all possible column offsets
        for (auto colOffset = 0; colOffset < _corShape[1]; ++colOffset) {
            // launch each kernel
            // N.B. kernel launch is an implicit barrier, so no need for any extra synchronization
            ::correlate <<<B,T,S>>> (
                                     dArena, dAverage,
                                     _refShape[0], _refCells,
                                     _tgtShape[0], _tgtCells,
                                     _corShape[0], rowOffset, colOffset,
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
    // stop the clock
    timer.stop();
    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while waiting for a kernel to finish: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // release device memory
        cudaFree(dCorrelation);
        // and bail
        return nullptr;
    }
    // report the timing
    tlog
        << pyre::journal::at(__HERE__)
        << "correlation kernel: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // transfer the data from the device
    // allocate room on the host
    auto hCorrelation = new cell_type[size];
    // start the clock
    timer.reset().start();
    // transfer the data
    status = cudaMemcpy(hCorrelation, dCorrelation, footprint, cudaMemcpyDeviceToHost);
    // stop the clock
    timer.stop();
    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while transferring the correlation matrix from the device: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // release device memory
        cudaFree(dCorrelation);
        // host memory
        delete [] hCorrelation;
        // and bail
        return nullptr;
    }
    // read the timer
    auto elapsed = timer.read();
    // compute the bandwidth
    auto bandwidth = footprint / elapsed; // in bytes/second
    // report the timing
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the correlation matrix from the device: " << 1e6 * elapsed << " μs"
        << " at " << (bandwidth/1024/1024/1024) << " Gb/s"
        << pyre::journal::endl;

    // clean up
    cudaFree(dCorrelation);

    // all done
    return hCorrelation;
}


// the correlation kernel
__global__ static
void
correlate(const cell_t * arena, // the dataspace
          const cell_t * average, // the hyper-grid of target amplitude averages
          std::size_t rdim, std::size_t rcells, // ref grid shape and size
          std::size_t tdim, std::size_t tcells, // tgt grid shape and size
          std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
          cell_t * correlation) {

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
    extern __shared__ cell_t scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // initialize my partial sum
    cell_t num = 0;

    // reference and target grids area interleaved; compute the stride
    std::size_t stride = rcells * tcells;

    // my {ref} starting point is column {t} of grid {b}
    auto ref = arena + b*stride + t;
    // my {tgt} starting point is column {t} of grid {b} at (rowOffset, colOffset)
    // cell_t * tgt = arena + b*stride + rcells + (rowOffset*tdim + colOffset) + t;
    // or, more simply
    auto tgt = ref + rcells + (rowOffset*tdim + colOffset);

    // run down the two columns
    for (std::size_t idx=0; idx < rdim; ++idx) {
        // fetch the ref value
        cell_t r = ref[idx*rdim];
        // fetch the tgt value
        cell_t t = tgt[idx*tdim];
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
