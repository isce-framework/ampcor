// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
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
// grab the cpu sat
#include <ampcor/correlators.h>

// the data grid
using grid_t = pyre::grid::simple_t<2, double, std::array<std::size_t,2>>;

// my reduction
__global__
static void sat(grid_t::cell_type * data, size_t nRows, size_t nCols,
                grid_t::cell_type * result);

// driver
int main() {
    // make a timer
    pyre::timer_t timer("ampcor.cuda.tile");

    // the number of data grids in this example mimic a reasonable correlation plan load
    int G = 120*40; // this is capacity on dgx (?)
    // dim
    grid_t::index_type::value_type dim = 32 + 128 + 32;
    // shape
    grid_t::shape_type shape {dim, dim};
    // make a grid
    grid_t grid { shape };
    // make a view over the grid
    auto view = grid.view();
    // unpack the shape
    size_t nRows = grid.layout().shape()[0];
    size_t nCols = grid.layout().shape()[1];
    // compute its footprint
    auto footprint = grid.layout().size() * sizeof(grid_t::cell_type);
    // pick a value
    grid_t::cell_type value = 1;
    // fill the grid
    std::fill(view.begin(), view.end(), value);

    // the first addressable grid cell
    grid_t::shape_type first {0ul, 0ul};
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
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (G*footprint/1024/1024) << " Mb for the grid data"
        << pyre::journal::endl;

    // find a spot
    grid_t::cell_type * dResult = nullptr;
    // allocate device memory for the results; we need as many slots as we have grids
    status = cudaMallocManaged(&dResult, G * footprint);
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
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (G*footprint/1024/1024) << " Mb for the sum tables"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // as many times as we have grids
    for (auto gid = 0; gid < G; ++gid) {
        // move the data
        status = cudaMemcpy(dGrid + gid*nRows*nCols,
                            grid.data(),
                            footprint,
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

    // each thread block takes care of one grid
    size_t B = G;
    // the number of threads is equal to the largest grid shape component, but really the grids
    // must be square
    size_t T = std::max(nRows, nCols);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // launch a kernel
    sat <<<B, T>>> (dGrid, nRows, nCols, dResult);
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

    // start the clock
    timer.reset().start();
    // here is the correct answer
    ampcor::correlators::sumarea_t<grid_t> gold(grid);
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "preparing the correct answer: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // now, harvest each sum table
    for (auto gid = 0; gid < G; ++gid) {
        // make a SAT
        grid_t sat { grid.layout() };
        // move the data over
        status = cudaMemcpy(sat.data(),
                            dResult + gid * sat.layout().size(),
                            footprint,
                            cudaMemcpyDeviceToHost);
        // check
        if (status != cudaSuccess) {
            // make a channel
            pyre::journal::error_t channel("ampcor.cuda.tile");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "while harvesting SAT #" << gid << " from the device: "
                << cudaGetErrorName(status) << " (" << status << ")"
                << pyre::journal::endl;
            // and bail
            return 1;
        }

        // verify the result
        for (auto idx : sat.layout()) {
            // expected
            auto expected = gold[idx];
            // actual
            auto actual = sat[idx];
            // if it's not what we expect
            if (actual != expected) {
                // make a channel
                pyre::journal::error_t error("ampcor.cuda.tile");
                // complain
                error
                    << pyre::journal::at(__HERE__)
                    << "sat[" << idx << "] = " << actual << " != " << expected
                    << pyre::journal::endl;
                // and bail
                // return 1;
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "harvesting and verifying the results: " << 1e6 * timer.read() << " μs"
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
sat(grid_t::cell_type * grid, size_t nRows, size_t nCols, grid_t::cell_type * result) {
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

    // first, each thread sweeps down its row storing the partial sums, followed by sweep down
    // it column

    // across the row
    // my starting point is row t of grid b
    std::size_t rowStart = b*nCols*nRows + t*nCols;
    // the stopping point can't go past the end of my row
    std::size_t rowStop = rowStart + nCols;

    // initialize the partial sum
    grid_t::cell_type sum = 0;
    // run
    for (auto slot = rowStart; slot < rowStop; ++slot) {
        // update the sum
        sum += grid[slot];
        // store the result
        result[slot] = sum;
    }

    // barrier: make sure everybody is done updating the result grid
    cta.sync();

    // down the column
    // my starting point is column t of grid b
    std::size_t colStart = b*nCols*nRows + t;
    // my end point can't go past the end of my grid
    std::size_t colStop = (b+1)*nCols*nRows;
    // re-initialize the partial sum
    sum = 0;
    // run
    for (auto slot=colStart; slot < colStop; slot += nCols) {
        // read the current value
        auto current = result[slot];
        // update the current value
        result[slot] += sum;
        // update the running sum
        sum += current;
    }

    // all done
    return;
}


// end of file
