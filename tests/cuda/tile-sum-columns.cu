// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// standard library
#include <algorithm>
#include <numeric>
// cuda
#include <cuda_runtime.h>
#include <cooperative_groups.h>
// support
#include <pyre/journal.h>
#include <pyre/grid.h>
#include <pyre/timers.h>

// the data grid
using grid_t = pyre::grid::simple_t<2, double>;
// the vector with the results
using result_t = pyre::grid::simple_t<1, grid_t::cell_type>;

// my reduction
__global__
static void sum(grid_t::cell_type * data, size_t nRows, size_t nCols,
                grid_t::cell_type * result);

// driver
int main() {
    // make a timer
    pyre::timer_t timer("ampcor.cuda.tile");

    // the number of data grids in this example
    int G = 1280; // this is capacity on dgx (?)
    // shape
    grid_t::shape_type shape {128, 128};
    // make a grid
    grid_t grid { shape };
    // make a view over the grid
    auto view = grid.view();
    // unpack the shape
    size_t nRows = grid.layout().shape()[0];
    size_t nCols = grid.layout().shape()[1];
    // compute its footprint
    auto footprint = grid.layout().size() * sizeof(grid_t::cell_type);
    // fill the grid
    for (auto idx : grid.layout()) {
        // with consecutive values
        grid[idx] = grid.layout().offset(idx);
    }

    // the first addressable grid cell
    grid_t::shape_type first {0, 0};
    // the last addressable grid cell
    grid_t::shape_type last = shape - grid_t::shape_type::fill(1);

    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.tile");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "grid:" << pyre::journal::newline
        << "  shape: " << grid.layout().shape()
        << pyre::journal::newline
        << "  packing: " << grid.layout().packing()
        << pyre::journal::newline
        << "  size: " << grid.layout().size() << " cells"
        << pyre::journal::newline
        << "  footprint: " << footprint << " bytes"
        << pyre::journal::newline
        << "grid[" << first <<"]: " << grid[first]
        << pyre::journal::newline
        << "grid[" << last << "]: " << grid[last]
        << pyre::journal::endl;

    // grab a device
    cudaError_t status = cudaSetDevice(0);
    // if anything went wrong
    if (status != cudaSuccess) {
        // make an error channel
        pyre::journal::error_t error("ampcor.cuda.tile");
        // show me
        error
            << pyre::journal::at(__HERE__)
            << "while grabbing a device: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // find a spot
    grid_t::cell_type * dGrid = nullptr;
    // allocate device memory for the grid payload
    status = cudaMallocManaged(&dGrid, G * footprint);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (G*footprint/1024/1024) << " Mb"
        << pyre::journal::endl;
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the grid payload: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // find a spot
    grid_t::cell_type * dResult = nullptr;
    // allocate device memory for the results; we need as many slots as we have grids
    status = cudaMallocManaged(&dResult, G * sizeof(grid_t::cell_type));
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the result: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // start the clock
    timer.reset().start();
    // as many times as we have grids
    for (auto gid = 0; gid < G; ++gid) {
        // move the data
        status = cudaMemcpy(dGrid + gid*nRows*nCols,
                            grid.data(),
                            nRows*nCols*sizeof(grid_t::cell_type),
                            cudaMemcpyHostToDevice);
        // check
        if (status != cudaSuccess) {
            // make a channel
            pyre::journal::error_t channel("ampcor.cuda.tile");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "while copying data from grid #" << gid << " to the device: "
                << cudaGetErrorName(status) << " (" << status << ")"
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "moving grid data to the device: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // figure out the job layout

    // each thread block takes care of one grid; each thread in the block handles a single
    // column, so we can coalesce the global memory reads. this implies that the number of
    // columns in the grid cannot exceed the maximum number of threads in a block
    // we only have one grid, so we only need one block
    size_t B = G;
    // the number of threads is equal to the number of columns
    size_t T = nCols;
    // each thread leaves behind the sum of its column in shared memory, so we need as many
    // {cell_type} slots as there are column in the grid; hence, the total amount of shared
    // memory we require is:
    size_t S = T * sizeof(grid_t::cell_type);

    //
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // launch a kernel
    sum <<<B, T, S>>> (dGrid, nRows, nCols, dResult);
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "kernel: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while waiting for a kernel to finish: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // the shape of the result vector
    result_t::shape_type rshape { G };
    // allocate room for the result on the host
    result_t result { rshape };
    // start the clock
    timer.reset().start();
    // move the answer over
    status = cudaMemcpy(result.data(),
                        dResult,
                        G*sizeof(result_t::cell_type),
                        cudaMemcpyDeviceToHost);
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "harvesting the results from the device: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;
    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while harvesting the sum from the device: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }

    // start the clock
    timer.reset().start();
    // the expected answer
    grid_t::cell_type expected = std::accumulate(view.begin(), view.end(), 0);
    // go through each one of the results
    for (auto idx : result.layout()) {
        // get the value
        auto value = result[idx];
        // if it's not what we expect
        if (value != expected) {
            // make a channel
            pyre::journal::error_t error("ampcor.cuda.tile");
            // complain
            error
                << pyre::journal::at(__HERE__)
                << "result[" << idx << "] = " << value << " != " << expected
                << pyre::journal::endl;
            // and bail
            return 1;
        }
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "verifying the results: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // cleanup
    cudaFree(dGrid);
    cudaFree(dResult);

    // all done
    return 0;
}


// the kernel
__global__
void
sum(grid_t::cell_type * grid, size_t nRows, size_t nCols, grid_t::cell_type * result) {
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
    extern __shared__ grid_t::cell_type scratch[];
    // get a handle to this thread block group
    cooperative_groups::thread_block cta = cooperative_groups::this_thread_block();

    // initialize my partial sum
    grid_t::cell_type sum = 0;

    // my starting point is column t of grid b
    std::size_t start = b*nCols*nRows + t;
    // my stopping point can't go past the end of my grid
    std::size_t stop = (b+1)*nCols*nRows;

    // run down my column
    for (std::size_t slot=start; slot<stop; slot += nCols) {
        // read the value from global memory and update my running total
        sum += grid[slot];
    }

    // save the partial sum
    scratch[t] = sum;

    // barrier: make sure everybody is done updating shared memory with their partial sum
    cta.sync();

    // now, do the reduction in shared memory
    // N.B.: we assume the warp size is 32; this will need updates if the warp size changes
    if (t < 64) {
        // pull a neighbor's value
        sum += scratch[t + 64];
        // and update my slot
        scratch[t] = sum;
    }

    // barrier: make sure everybody is done updating shared memory with their partial sum
    cta.sync();

    // we are now within a warp
    if (t < 32) {
        // get a handle to the active thread group
        cooperative_groups::coalesced_group active = cooperative_groups::coalesced_threads();
        // pull the partial result from the second warp
        sum += scratch[t + 32];
        // the power-of-2 threads
        for (int offset = 16; offset > 0; offset >>= 1) {
            // reduce using {shuffle}
            sum += active.shfl_down(sum, offset);
        }
    }

    // the master thread
    if (t == 0) {
        // writes the sum to the result vector
        result[b] = sum;
    }

    // all done
    return;
}


// end of file
