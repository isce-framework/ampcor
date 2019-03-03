// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
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
avg(const cell_t * sat,
      std::size_t tiles, std::size_t tgtDim, std::size_t corDim,
      cell_t * avg);


// implementation

// precompute the amplitude averages for all possible placements of the search tile within the
// target search window for all pairs in the plan. we allocate room for {_pairs}*{_corCells}
// floating point values and use the precomputed SAT tables resident on the device.
//
// the SAT tables require a slice and produce the sum of the values of cells within the slice
// in no more than four memory accesses per search tile; there are boundary cases to consider
// that add a bit of complexity to the implementation; the boundary cases could have been
// trivialized using ghost cells around the search window boundary, but the memory cost is high
auto
ampcor::cuda::correlators::CUDA::
_newAverageTargetAmplitudes(const cell_type * dSAT) const -> cell_type *
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::info_t tlog("ampcor.cuda.timings");

    // allocate room for the table of amplitude averages
    // pick a spot
    cell_type * dAverage;
    // the total number of cells in the amplitude hyper-grid
    auto size = _pairs * _corCells;
    // the amount of memory needed to store them
    auto footprint = size * sizeof(cell_type);
    // allocate some device memory
    cudaError_t status = cudaMallocManaged(&dAverage, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the table of target amplitude averages: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        return nullptr;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << footprint << " bytes for the target amplitude averages at "
        << dAverage
        << pyre::journal::endl;
    // start the clock
    timer.reset().start();
    // launch blocks of 256 threads
    auto T = 256;
    // in as many blocks as it takes to handle all pairs
    auto B = _pairs % T ? (_pairs/T + 1) : _pairs/T;
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T
        << " threads each to handle the " << _pairs
        << " UL corners of the hyper-grid of target amplitude averages"
        << pyre::journal::endl;
    // launch the kernels
    avg <<<B,T>>> (dSAT, _pairs, _tgtShape[0], _corShape[0], dAverage);
    // wait for the kernels to finish
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
            << "while computing the average amplitudes of all possible search window placements: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // release device memory
        cudaFree(dAverage);
        // and bail
        return nullptr;
    }
    // report the timing
    tlog
        << pyre::journal::at(__HERE__)
        << "averaging kernel: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // hand the hyper-grid memory to the caller
    return dAverage;
}


// the SAT generation kernel
__global__ static
void
avg(const cell_t * dSAT,
      std::size_t tiles,     // the total number of target tiles
      std::size_t tgtDim,    // the shape of each target tile
      std::size_t corDim,    // the shape of each grid
      cell_t * dAverage)
{
    // build the workload descriptors
    // global
    // std::size_t B = gridDim.x;    // number of blocks
    std::size_t T = blockDim.x;      // number of threads per block
    // std::size_t W = B*T;          // total number of workers
    // local
    std::size_t b = blockIdx.x;      // my block id
    std::size_t t = threadIdx.x;     // my thread id within my block
    std::size_t w = b*T + t;         // my worker id

    // if my worker id exceeds the number of cells that require update
    if (w >= tiles) {
        // nothing for me to do
        return;
    }

    // locate the beginning of my SAT table
    auto sat = dSAT + w*tgtDim*tgtDim;
    // locate the beginning of my table of averages
    auto avg = dAverage + w*corDim*corDim;

    // go through all possible row offsets
    for (auto row = 0; row < corDim; ++row) {
        // the row limit of the tile
        auto rowMax = row + corDim - 1;
        // go through all possible column offsets
        for (auto col = 0; col < corDim; ++col) {
            // the column limit of the tile
            auto colMax = col + corDim - 1;

            // initialize the sum by reading the bottom right corner; it's guaranteed to be
            // within the SAT
            auto sum = sat[rowMax*tgtDim + colMax];

            // if the slice is not top-aligned
            if (row > 0) {
                // subtract the value from the upper right corner
                sum -= sat[(row-1)*tgtDim + colMax];
            }

            // if the slice is not left-aligned
            if (col > 0) {
                // subtract the value of the upper left corner
                sum -= sat[rowMax*tgtDim + (col - 1)];
            }

            // if the slice is not aligned with the upper left corner
            if (row > 0 && col > 0) {
                // restore its contribution to the sum
                sum += sat[(row-1)*tgtDim + (col-1)];
            }

            // compute the average value and store it
            avg[row*corDim + col] = sum / (tgtDim*tgtDim);
        }
    }

    // all done
    return;
}


// end of file
