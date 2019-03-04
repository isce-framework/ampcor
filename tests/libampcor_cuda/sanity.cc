// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>
// support
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

    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "sanity check for the cuda ampcor task manager"
        << pyre::journal::endl;

    // the number of pairs
    correlator_t::size_type pairs = 1;
    // the reference shape
    correlator_t::shape_type refShape = { 128, 128};
    // the search window shape
    correlator_t::shape_type tgtShape = { 192, 192 };

    // start the clock
    timer.reset().start();
    // make a correlator
    correlator_t c(pairs, refShape, tgtShape);
    // stop the clock
    timer.stop();

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "instantiating the manager: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // all done
    return 0;
}

// end of file
