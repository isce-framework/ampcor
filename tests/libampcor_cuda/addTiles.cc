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
// support
#include <pyre/grid.h>
#include <pyre/journal.h>
#include <pyre/timers.h>
// ampcor
#include <ampcor_cuda/correlators.h>

// type aliases
// my raster type
using slc_t = pyre::grid::simple_t<2, std::complex<float>>;
// the correlator
using correlator_t = ampcor::cuda::correlators::interim_t<slc_t>;

// driver
int main() {
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
    int refExt = 4;
    // the margin around the reference tile
    int margin = 2;
    // therefore, the target tile extent
    int tgtExt = refExt + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    int placements = 2*margin + 1;
    // the number of pairs
    slc_t::size_type pairs = placements*placements;

    // the reference shape
    slc_t::shape_type refShape = {refExt, refExt};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtExt, tgtExt};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // start the clock
    timer.start();
    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.start();
    // make a reference raster
    slc_t ref(refLayout);
    // fill it with ones
    std::fill(ref.view().begin(), ref.view().end(), 1);
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refExt, j+refExt});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with ones
            std::fill(view.begin(), view.end(), 1);

            // show me
            channel << "tgt[" << i << "," << j << "]:" << pyre::journal::newline;
            for (auto idx=0; idx<tgtExt; ++idx) {
                for (auto jdx=0; jdx<tgtExt; ++jdx) {
                    channel << tgt[{idx, jdx}] << " ";
                }
                channel << pyre::journal::newline;
            }
            channel << pyre::journal::endl;

            // compute the pair id
            int pid = i*placements + j;
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
        << "creating reference dataset: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // show me
    c.dump();

    // all done
    return 0;
}

// end of file
