/// -*- C++ -*-

// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// standard library
#include <cmath>
#include <algorithm>
#include <numeric>
// cuda
#include <cuda_runtime.h>
#include <cufft.h>
#include <cooperative_groups.h>
// support
#include <pyre/journal.h>
#include <pyre/grid.h>
#include <pyre/timers.h>
// grab the cpu sat
#include <ampcor/correlators.h>

// the data grid
using grid_t = pyre::grid::simple_t<2, cufftDoubleComplex, std::array<std::size_t,2>>;


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
    std::size_t P = px*py;
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "plan: " << px << "x" << py << " tiles, for a total of " << P << " pairings"
        << pyre::journal::endl;

    // the grid dimension
    grid_t::index_type::value_type dim = 128;
    // the shape
    grid_t::shape_type shape {dim, dim};
    // declare the grid
    grid_t grid { shape };
    // compute the grid size
    grid_t::size_type size = grid.layout().size();
    // and its memory footprint
    grid_t::size_type footprint = size * sizeof(grid_t::cell_type);

    // fill the grid with values by quarters
    // at the top left
    for (grid_t::size_type i = 0; i < dim/2; ++i) {
        for (grid_t::size_type j = 0; j < dim/2; ++j) {
            // put a constant value
            grid[{i,j}] = {.5, 0};
        }
    }
    // at the top right
    for (grid_t::size_type i = 0; i < dim/2; ++i) {
        for (grid_t::size_type j = dim/2; j < dim; ++j) {
            // put a constant value
            grid[{i,j}] = {0, 1};
        }
    }
    // bottom left
    for (grid_t::size_type i = dim/2; i < dim; ++i) {
        for (grid_t::size_type j = 0; j < dim/2; ++j) {
            // put a constant value
            grid[{i,j}] = {1, 0};
        }
    }
    // bottom right
    for (grid_t::size_type i = dim/2; i < dim; ++i) {
        for (grid_t::size_type j = dim/2; j < dim; ++j) {
            // put a constant value
            grid[{i,j}] = {1, 1};
        }
    }

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

    // the amount of memory we need on the device
    grid_t::size_type dFootprint = P * footprint;
    // set aside a spot for the data
    grid_t::cell_type * dGrid = nullptr;
    // allocate device memory
    status = cudaMallocManaged(&dGrid, dFootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the tile"
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // zero it out
    status = cudaMemset(dGrid, 0, dFootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while initializing device memory for the tiles: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (dFootprint) << " bytes for the tile data at "
        << dGrid
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // move the data to the device
    for (std::size_t pid = 0ul; pid < P; ++pid) {
        // copy the data to the matching arena slot
        status = cudaMemcpy(dGrid + pid*size, grid.data(), footprint, cudaMemcpyHostToDevice);
        // check
        if (status != cudaSuccess) {
            // make a channel
            pyre::journal::error_t channel("ampcor.cuda.tile");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "while copying data to device grid #" << pid << ": "
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
        << "moving data to the device: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // room for the results
    grid_t::cell_type * rGrid = nullptr;
    // allocate device memory
    status = cudaMallocManaged(&rGrid, dFootprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the FFT results: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // zero it out
    status = cudaMemcpy(rGrid, dGrid, dFootprint, cudaMemcpyDeviceToDevice);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while initializing device memory for the FFT results: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (dFootprint) << " bytes for the FFT results at "
        << dGrid
        << pyre::journal::endl;

    // the FFT plan characteristics
    int rank = 2;
    int ranks[] = { (int)dim/2, (int)dim/2 };

    int inembed[] = { (int)dim, (int)dim };
    int istride = 1;
    int idist = size;

    int onembed[] = { (int)dim, (int)dim };
    int ostride = 1;
    int odist = size;

    // the estimated footprint
    std::size_t planEstimatedFootprint = 0;
    // start the clock
    timer.reset().start();
    // estimate
    cufftResult_t statusFFT = cufftEstimateMany(rank, ranks,
                                                inembed, istride, idist,
                                                onembed, ostride, odist,
                                                CUFFT_Z2Z,
                                                P,
                                                &planEstimatedFootprint);
    // if something went wrong
    if (statusFFT != CUFFT_SUCCESS) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while estimating the size of the FFT plan: error " << statusFFT
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "estimated FFT footprint: " << planEstimatedFootprint << " bytes"
        << pyre::journal::newline
        << "FFT plan estimation: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // the FFT plan
    cufftHandle plan;
    // instantiate
    statusFFT = cufftPlanMany(&plan,
                              rank, ranks,
                              inembed, istride, idist,
                              onembed, ostride, odist,
                              CUFFT_Z2Z,
                              P);
    // if something went wrong
    if (statusFFT != CUFFT_SUCCESS) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while creating the FFT plan: error " << statusFFT
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "FFT plan creation: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // execute the plan
    statusFFT = cufftExecZ2Z(plan, dGrid, rGrid, CUFFT_FORWARD);
    // if something went wrong
    if (statusFFT != CUFFT_SUCCESS) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while executing the FFT plan: error " << statusFFT
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // wait for the device to finish
    status = cudaDeviceSynchronize();
    // stop the clock
    timer.stop();
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while executing the FFT plan: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "FFT plan execution: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // allocate room for the results back on the host
    grid_t::cell_type *results = new grid_t::cell_type[P*size];
    // move the data back to the host
    status = cudaMemcpy(results, rGrid, dFootprint, cudaMemcpyDeviceToHost);
    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda.tile");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while copying results back to the host: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        return 1;
    }
    // stop the clock
    timer.stop();
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "moving results back to the host: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // our error tolerance
    double tolerance = 0;
    // show me
    for (std::size_t pid = 0ul; pid < P; ++pid) {
        // in the upper left corner, we should have the FFT of our constant
        for (std::size_t i = 0ul; i < dim/2; ++i) {
            for (std::size_t j = 0ul; j < dim/2; ++j) {
                // get the value
                grid_t::cell_type value = results[pid*dim*dim + i*dim + j];
                // unpack
                double re = value.x;
                double im = value.y;

                // at the origin
                if (i == 0 && j == 0) {
                    // the correct answer is a non-zero real part and a vanishing imaginary part
                    if (std::abs(re) < tolerance || std::abs(im) > tolerance) {
                        // make a channel
                        pyre::journal::error_t error("ampcor.cuda.tile");
                        // complain
                        error
                            << pyre::journal::at(__HERE__)
                            << "top corner: result[" << pid << "; " << i << "," << j << "] = "
                            << "(" << re << ", " << im << ") != δ(0)"
                            << pyre::journal::endl;
                        // and bail
                        return 1;
                    }
                }
                // everywhere else
                else {
                    // both real and imaginary parts vanish
                    if (std::abs(re) > tolerance || std::abs(im) > tolerance) {
                        // make a channel
                        pyre::journal::error_t error("ampcor.cuda.tile");
                        // complain
                        error
                            << pyre::journal::at(__HERE__)
                            << "top left: result[" << pid << "; " << i << "," << j << "] = "
                            << "(" << re << ", " << im << ") != 0"
                            << pyre::journal::endl;
                        // and bail
                        return 1;
                    }
                }
            }
        }
        // in the upper right hand corner, we should have {0, 1}
        for (grid_t::size_type i = 0ul; i < dim/2; ++i) {
            for (grid_t::size_type j = dim/2; j < dim; ++j) {
                // get the value
                grid_t::cell_type value = results[pid*dim*dim + i*dim + j];
                // unpack
                double re = value.x;
                double im = value.y;
                // check
                if (re != 0 && im != 1) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda.tile");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "top right: result[" << pid << "; " << i << "," << j << "] = "
                        << "(" << re << ", " << im << ") != (0,1)"
                        << pyre::journal::endl;
                    // and bail
                    return 1;
                }
            }
        }
        // in the bottom left, we should have {1, 0}
        for (grid_t::size_type i = dim/2; i < dim; ++i) {
            for (grid_t::size_type j = 0ul; j < dim/2; ++j) {
                // get the value
                grid_t::cell_type value = results[pid*dim*dim + i*dim + j];
                // unpack
                double re = value.x;
                double im = value.y;
                // check
                if (re != 1 && im != 0) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda.tile");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "bottom left: result[" << pid << "; " << i << "," << j << "] = "
                        << "(" << re << ", " << im << ") != (0,1)"
                        << pyre::journal::endl;
                    // and bail
                    return 1;
                }
            }
        }
        // in the bottom right, we should have {1, 1}
        for (grid_t::size_type i = dim/2; i < dim; ++i) {
            for (grid_t::size_type j = dim/2; j < dim; ++j) {
                // get the value
                grid_t::cell_type value = results[pid*dim*dim + i*dim + j];
                // unpack
                double re = value.x;
                double im = value.y;
                // check
                if (re != 1 && im != 1) {
                    // make a channel
                    pyre::journal::error_t error("ampcor.cuda.tile");
                    // complain
                    error
                        << pyre::journal::at(__HERE__)
                        << "bottom right: result[" << pid << "; " << i << "," << j << "] = "
                        << "(" << re << ", " << im << ") != (1,1)"
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
        << "checking results back at the host: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // clean up
    cudaFree(dGrid);
    cudaFree(rGrid);
    cufftDestroy(plan);
    delete [] results;

    // all done
    return 0;
}


// end of file
