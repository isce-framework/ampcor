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


// the SAT generation kernel
__global__
static void
sat(const cell_t * dArena,
    std::size_t stride, std::size_t rcells, std::size_t tdim,
    cell_t * dSAT);


// implementation
auto
ampcor::cuda::correlators::CUDA::
_newSAT(const cell_type * dArena) const -> cell_type *
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::info_t tlog("ampcor.cuda.timings");

    // build sum area tables for the target tiles
    // find a spot
    cell_type * dSAT = nullptr;
    // the total number of cells in the SAT hyper-grid
    auto size = _pairs * _tgtCells;
    // the amount of memory needed to store them
    auto footprint = size * sizeof(cell_type);
    // allocate memory
    cudaError_t status = cudaMallocManaged(&dSAT, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the sum area tables: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        return nullptr;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << footprint << " bytes for the sum area tables at "
        << dSAT
        << pyre::journal::endl;

    // to compute the SAT for each target tile, we launch as many thread blocks as there are
    // target tiles
    std::size_t B = _pairs;
    // the number of threads per block is determined by the shape of the target tile
    std::size_t T = _tgtShape[1];
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to compute SATs for the target tiles"
        << pyre::journal::endl;
    // start the clock
    timer.reset().start();
    // launch the SAT kernel
    sat <<<B,T>>> (dArena, _refCells*_tgtCells, _refCells, _tgtShape[0], dSAT);
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // stop the clock
    timer.stop();
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while computing the sum area tables: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // release device memory
        cudaFree(dSAT);
        // bail
        return nullptr;
    }
    // report the timing
    tlog
        << pyre::journal::at(__HERE__)
        << "SAT generation kernel: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // hand the SAT hyper-grid memory to the caller
    return dSAT;
}


// the SAT generation kernel
__global__ static
void
sat(const cell_t * dArena,
    std::size_t stride, std::size_t rcells, std::size_t tdim,
    cell_t * dSAT)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    // std::size_t T = blockDim.x;   // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    // std::size_t w = b*T + t;      // my worker id

    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // on the first pass, each thread sweeps across its row
    // on a second pass, each thread sweeps down its column

    // across the row
    // my starting point for reading data is row {t} of tile {b} in the arena
    std::size_t read = b*stride + rcells + t*tdim;
    // my starting point for writing data is row {t} of tile {b} in the SAT area
    std::size_t write = b*tdim*tdim + t*tdim;

    // initialize the partial sum
    cell_t sum = 0;

    // run across the row
    for (auto slot = 0; slot < tdim; ++slot) {
        // update the sum
        sum += dArena[read + slot];
        // store the result
        dSAT[write + slot] = sum;
    }

    // barrier: make sure everybody is done updating the SAT
    cta.sync();

    // down the column of the SAT table itself
    // my starting point is column {t} of tile {b}
    std::size_t colStart = b*tdim*tdim + t;
    // and my stopping point can't go past the end of my tile
    std::size_t colStop = (b+1)*tdim*tdim;
    // reinitialize the partial sum
    sum = 0;
    // run
    for (auto slot=colStart; slot < colStop; slot += tdim) {
        // read the current value and save it
        auto current = dSAT[slot];
        // update the current value with the running sum
        dSAT[slot] += sum;
        // update the running sum for the next guy
        sum += current;
    }

    // all done
    return;
}


// end of file
