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
#include <cmath>
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
static void correlate(grid_t::cell_type * ref, std::size_t rdim,
                      grid_t::cell_type * tgt, std::size_t tdim,
                      std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
                      grid_t::cell_type * correlation);


// driver
int main(int argc, char *argv[]) {
    // the plan shape
    int px = 120;
    int py = 40;

    // unpack command line arguments
    if (argc > 1) {
        px = std::atoi(argv[1]);
    }
    if (argc > 2) {
        py = std::atoi(argv[2]);
    }

    // make a timer
    pyre::timer_t timer("ampcor.cuda.tile");
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda.tile");

    //  the plan
    int P = px*py;
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "plan: " << px << "x" << py << " tiles, for a total of " << P << " pairings"
        << pyre::journal::endl;

    // the reference grid dimension
    grid_t::index_type::value_type rdim = 128;
    // the target grid shape includes a margin
    grid_t::index_type::value_type margin = 32;
    // so here is its dimension
    grid_t::index_type::value_type tdim = margin + rdim + margin;
    // the shapes
    grid_t::shape_type rshape {rdim, rdim};
    grid_t::shape_type tshape {tdim, tdim};
    grid_t::shape_type cshape { margin+1, margin+1 };
    // declare the grids
    grid_t ref { rshape };
    grid_t tgt { tshape };
    grid_t cor { cshape };

    // the sizes
    grid_t::size_type rsize = ref.layout().size();
    grid_t::size_type tsize = tgt.layout().size();
    grid_t::size_type csize = cor.layout().size();
    // memory footprints
    grid_t::size_type rfootprint = rsize * sizeof(grid_t::cell_type);
    grid_t::size_type tfootprint = tsize * sizeof(grid_t::cell_type);
    grid_t::size_type cfootprint = csize * sizeof(grid_t::cell_type);

    // pick a value for the reference grid payload
    grid_t::cell_type value = 1;
    // and a value for the mask around the border
    grid_t::cell_type mask = 0;

    // fill the reference grid the easy way
    for (auto idx : ref.layout()) {
        // by setting all slots to {value}
        ref[idx] = value;
    }
#if 0
    // show me
    channel << pyre::journal::at(__HERE__);
    // go through the ref slots
    for (auto idx : ref.layout()) {
        // show me the contents
        channel << "ref[" << idx << "] = " << ref[idx] << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;
#endif

    // initialize the target grid
    for (auto idx : tgt.layout()) {
        // by setting all slots to {mask}
        tgt[idx] = mask;
    }
    // turn the margin into an index
    grid_t::index_type mindex { margin, margin };
    // specify a region in the interior
    grid_t::index_type start = tgt.layout().low() + mindex;
    grid_t::index_type end = tgt.layout().high() - mindex;
    // fill the interior
    for (auto idx : tgt.layout().slice(start, end)) {
        // with {value}
        tgt[idx] = value;
    }
#if 0
    // show me
    channel << pyre::journal::at(__HERE__);
    // go through the ref slots
    for (auto idx : tgt.layout()) {
        // show me the contents
        channel << "tgt[" << idx << "] = " << tgt[idx] << pyre::journal::newline;
    }
    // flush
    channel << pyre::journal::endl;
#endif

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

    // find a spot for the reference grids
    grid_t::cell_type * dRef = nullptr;
    // allocate device memory for the grid payload
    status = cudaMallocManaged(&dRef, P * rfootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the reference payload: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (P*rfootprint) << " bytes for the grid data at "
        << dRef
        << pyre::journal::endl;

    // find a spot for the target grids
    grid_t::cell_type * dTgt = nullptr;
    // allocate device memory for the grid payload
    status = cudaMallocManaged(&dTgt, P * tfootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the target payload: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (P*tfootprint) << " bytes for the grid data at "
        << dTgt
        << pyre::journal::endl;

    // make room for the answer
    grid_t::cell_type * dCorrelation = nullptr;
    // allocate device memory for the grid payload
    status = cudaMallocManaged(&dCorrelation, P * cfootprint);
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
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << P * cfootprint
        << " bytes for the correlation matrix at "
        << dCorrelation
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // move the data to the device
    for (auto pid = 0; pid < P; ++pid) {
        // first the reference grid
        status = cudaMemcpy(dRef + pid*rsize,
                            ref.data(),
                            rfootprint,
                            cudaMemcpyHostToDevice);
        // check
        if (status != cudaSuccess) {
            // make a channel
            pyre::journal::error_t channel("ampcor.cuda.tile");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "while copying data from reference grid #" << pid << " to the device: "
                << cudaGetErrorName(status) << " (" << status << ")"
                << pyre::journal::endl;
            // and bail
            return 1;
        }
        // then the target grid
        status = cudaMemcpy(dTgt + pid*tsize,
                            tgt.data(),
                            tfootprint,
                            cudaMemcpyHostToDevice);
        // check
        if (status != cudaSuccess) {
            // make a channel
            pyre::journal::error_t channel("ampcor.cuda.tile");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "while copying data from target grid #" << pid << " to the device: "
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
        << "moving reference and target data to the device: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;


    // figure out the job layout
    // each thread block takes care of one grid pair
    std::size_t B = P;
    // the number of threads is equal to the number of rows/cols in the reference grid
    std::size_t T = rdim;
    // each thread stores its partial sum in shared memory, so we need as much shared memory
    // per block as there are columns in the {ref} grid
    std::size_t S = T * sizeof(grid_t::cell_type);
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "launching " << B << " blocks of " << T << " threads each"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // go through all possible row offsets
    for (auto rowOffset = 0; rowOffset < margin + 1; ++rowOffset) {
        // and all possible column offsets
        for (auto colOffset = 0; colOffset < margin + 1; ++colOffset) {
            // launch a kernel
            // N.B.: kernel launch is an implicit barrier, so no need for extra synchronization
            correlate <<<B, T, S>>> (dRef, rdim, dTgt, tdim,
                                     margin, rowOffset, colOffset,
                                     dCorrelation);
        }
    }
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

    // harvest the results
    timer.reset().start();
    // make room on the host for the correlation grids
    grid_t::cell_type * correlation = new grid_t::cell_type[P * csize];
    // move the correlation results
    status = cudaMemcpy(correlation,
                        dCorrelation,
                        P * cfootprint,
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

    // verify
    timer.reset().start();
    // go through all the pairs
    for (auto pid = 0; pid < P; ++pid) {
        // the starting point of this result grid
        std::size_t start = pid * (margin+1)*(margin+1);
        // go through all row offsets
        for (auto rowOffset = 0; rowOffset < margin+1; ++rowOffset) {
            // and all column offsets
            for (auto colOffset = 0; colOffset < margin+1; ++colOffset) {
                // compute the number of times {value} shows up in this sub-grid
                auto live = (rdim-std::abs(margin-rowOffset))*(rdim-std::abs(margin-colOffset));
                // therefore, the expected result is
                auto expected = live * value;
                // get the actual
                auto actual = correlation[start + rowOffset*(margin+1) + colOffset];
                // if they don't match
                if (expected != actual) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda.tile");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "cor["
                        << pid << "," << rowOffset << "," << colOffset
                        << "] = " << actual << " != " << expected
                        << pyre::journal::endl;
                    // and bail
                    return 1;
                }
            }
        }
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "verifying the results: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // clean up
    cudaFree(dRef);
    cudaFree(dTgt);
    cudaFree(dCorrelation);
    delete [] correlation;

    // all done
    return 0;
}


// the kernel
__global__
static void correlate(grid_t::cell_type * ref, std::size_t rdim, // ref grid data and shape
                      grid_t::cell_type * tgt, std::size_t tdim, // tgt grid data and shape
                      std::size_t margin, std::size_t rowOffset, std::size_t colOffset,
                      grid_t::cell_type * correlation) {

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
    grid_t::cell_type num = 0;

    // my ref starting point is column t of grid b
    std::size_t rstart = b*rdim*rdim + t;
    // my tgt starting point is column t of grid b at (rowOffset, colOffset)
    std::size_t tstart = (b*tdim*tdim) + (rowOffset*tdim + colOffset) + t;

    // run down the two columns
    for (std::size_t idx=0; idx < rdim; ++idx) {
        // fetch the ref value
        grid_t::cell_type r = ref[rstart + idx*rdim];
        // fetch the tgt value
        grid_t::cell_type t = tgt[tstart + idx*tdim];
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
        std::size_t slot = b*(margin+1)*(margin+1) + rowOffset*(margin+1) + colOffset;
        // writes the sum to the result vector
        correlation[slot] = num;
    }

    // all done
    return;
}
// end of file
