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
#include <numeric>
// cuda
#include <cuda_runtime.h>
// support
#include <pyre/grid.h>
#include <pyre/journal.h>
#include <pyre/timers.h>
// ampcor
#include <ampcor_cuda/correlators.h>

// type aliases
// my value type
using value_type = float;
// my raster type
using slc_t = pyre::grid::simple_t<2, std::complex<value_type>>;
// the correlator
using correlator_t = ampcor::cuda::correlators::interim_t<slc_t>;

// driver
int main() {
    // number of gigabytes per byte
    const auto Gb = 1.0/(1024*1024*1024);

    // make a timer
    pyre::timer_t timer("ampcor.cuda.sanity");
    // make a channel for reporting the timings
    pyre::journal::info_t tlog("ampcor.cuda.tlog");

    // make a channel for logging progress
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "setting up the correlation plan with the cuda ampcor task manager"
        << pyre::journal::endl;

    // the reference tile extent
    int refExt = 128;
    // the margin around the reference tile
    int margin = 32;
    // therefore, the target tile extent
    auto tgtExt = refExt + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;

    // the number of pairs
    auto pairs = placements*placements;
    // the number of cells in each pair
    auto cellsPerPair = refExt*refExt + tgtExt*tgtExt;

    // the total number of cells
    auto cells = pairs * (refExt*refExt + tgtExt*tgtExt);

    // the reference shape
    slc_t::shape_type refShape = {refExt, refExt};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtExt, tgtExt};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // compute the pair id
            int pid = i*placements + j;

            // make a reference raster
            slc_t ref(refLayout);
            // fill it with the pair id
            std::fill(ref.view().begin(), ref.view().end(), pid);

            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refExt, j+refExt});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "creating reference dataset: " << 1e3 * timer.read() << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // push the data to the device
    auto cArena = c._push();
    // stop the clock
    timer.stop();
    // get the duration
    auto wDuration = timer.read();
    // get the payload
    auto wFootprint = cells * sizeof(slc_t::cell_type);
    // compute the transfer rate in Gb/s
    auto wRate = wFootprint / wDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the dataset to the device: " << 1e3 * wDuration << " ms"
        << ", at " << wRate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena);
    // stop the clock
    timer.stop();
    // get the duration
    auto duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing amplitudes of the signal tiles: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // subtract the tile mean from each pixel
    c._zeroMean(rArena);
    // stop the clock
    timer.stop();
    // get the duration
    duration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "converting reference tiles to zero mean: " << 1e3 * duration << " ms"
        << pyre::journal::endl;

    // make room for the results
    auto * results = new value_type[cells];
    // compute the result footprint
    auto rFootprint = cells * sizeof(value_type);
    // start the clock
    timer.reset().start();
    // copy the results over
    cudaError_t status = cudaMemcpy(results, rArena, rFootprint, cudaMemcpyDeviceToHost);
    // stop the clock
    timer.stop();
    // if something went wrong
    if (status != cudaSuccess) {
        // get the error description
        std::string description = cudaGetErrorName(status);
        // make a channel
        pyre::journal::error_t error("ampcor.cuda");
        // complain
        error
            << pyre::journal::at(__HERE__)
            << "while retrieving the results: "
            << description << " (" << status << ")"
            << pyre::journal::endl;
        // and bail
        throw std::runtime_error(description);
    }

    // get the duration
    auto rDuration = timer.read();
    // compute the transfer rate
    auto rRate = rFootprint / rDuration * Gb;
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "moving the results to the host: " << 1e3 * rDuration << " ms"
        << ", at " << rRate << " Gb/s"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // verify
    for (auto pid = 0; pid < pairs; ++pid) {
        // compute the starting address of this tile
        auto mem = results + pid * cellsPerPair;
        // compute the mean of the tile
        auto mean = std::accumulate(mem, mem+refExt*refExt, 0.0) / (refExt*refExt);
        // verify it's near zero
        if (std::abs(mean) > std::numeric_limits<float>::epsilon()) {
            // make a channel
            pyre::journal::error_t error("ampcor.cuda");
            // complain
            channel
                << pyre::journal::at(__HERE__)
                << "mismatch at tile " << pid << ": " << mean << " != 0"
                << pyre::journal::endl;
            // bail
            break;
        }
    }
    // stop the clock
    timer.stop();
    // get the duration
    auto vDuration = timer.read();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "verifying results at the host: " << 1e3 * vDuration << " ms"
        << pyre::journal::endl;

    // if the debug channel is active
    if (channel) {
        // dump the resulting pairs
        for (auto pid = 0; pid < pairs; ++pid) {
            // sign in
            channel
                << pyre::journal::at(__HERE__)
                << "--------------------"
                << pyre::journal::newline
                << "pair " << pid << ":"
                << pyre::journal::newline;

            channel << "zero-mean reference:" << pyre::journal::newline;
            // the amplitude of the reference tile
            for (auto idx=0; idx < refExt; ++idx) {
                for (auto jdx=0; jdx < refExt; ++jdx) {
                    channel << results[pid*cellsPerPair + idx*refExt + jdx] << " ";
                }
                channel << pyre::journal::newline;
            }
        }
        channel << pyre::journal::endl;
    }

    // clean up
    cudaFree(cArena);
    cudaFree(rArena);
    delete [] results;

    // all done
    return 0;
}

// end of file
