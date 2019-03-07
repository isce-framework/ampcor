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
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <exception>
// pyre
#include <pyre/journal.h>
#include <pyre/timers.h>
#include <pyre/grid.h>
// cuda
#include <cuda_runtime.h>
// pull the local declarations
#include "CUDA.h"


// interface
void
ampcor::cuda::correlators::CUDA::
addReferenceTile(size_type pid, const slc_type & slc, slice_type slice)
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::debug_t tlog("ampcor.cuda.timings");

    // start the clock
    timer.reset().start();
    // compute the starting point of the reference tile that corresponds to this pair id
    cell_type * support = _hArena + pid*(_refCells + _tgtCells);
    // adapt it into a grid
    gview_type tile { {_refShape, slc.layout().packing()} , support };
    // build a view over the entire thing
    auto view = tile.view();

    // build a view to the reference raster that is limited to the supplied {slice}
    auto ref = slc.view(slice);

    // make a function that computes the magnitude of complex numbers
    auto magnitude = [] (slc_type::pixel_type pxl) -> cell_type
                     { return std::abs(pxl); };
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "pair #" << pid << ": reference tile start up: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // initialize the tile by applying {magnitude} to every cell in {ref} and storing the
    // result in {view}
    std::transform(ref.begin(), ref.end(), view.begin(), magnitude);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "reading reference tile, computing amplitudes, and storing: "
        << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // compute the sum of the tile amplitudes
    auto sum = std::accumulate(view.begin(), view.end(), 0);
    // compute the average value
    auto avg = sum / _refCells;
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "computing the amplitude average: "
        << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // build a function that subtracts the average from every amplitude
    auto rel = [avg] (cell_type cell) -> cell_type
               { return cell - avg; };

    // start the clock
    timer.reset().start();
    // subtract this value from every cell in the reference tile
    std::for_each(view.begin(), view.end(), rel);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "subtracting the average amplitude in the reference tile: "
        << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "adding reference tile #" << pid << ":" << pyre::journal::newline
        << "    pair id: " << pid << pyre::journal::newline
        << "    slice:" << pyre::journal::newline
        << "        from: (" << slice.low() << ")" << pyre::journal::newline
        << "        to: (" << slice.high() << ")" << pyre::journal::newline
        << "    support:" << pyre::journal::newline
        << "        anchor: " << _hArena << pyre::journal::newline
        << "        offset: " << pid*(_refCells + _tgtCells) << pyre::journal::newline
        << "        support: " << support << pyre::journal::newline
        << "        extent: " << tile.layout().size() << " cells" << pyre::journal::newline
        << "        footprint: " << sizeof(cell_type) * tile.layout().size() << " bytes"
        << pyre::journal::endl;

    // all done
    return;
}

void
ampcor::cuda::correlators::CUDA::
addTargetTile(size_type pid, const slc_type & slc, slice_type slice)
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::debug_t tlog("ampcor.cuda.timings");

    // start the clock
    timer.reset().start();
    // compute the starting point of the target tile that corresponds to this pair id
    cell_type * support = _hArena + pid*(_refCells + _tgtCells) + _refCells;
    // adapt it into a grid
    gview_type tile { {_tgtShape, slc.layout().packing()} , support };
    // build a view over the entire thing
    auto view = tile.view();

    // build a view to the target raster that is limited to the supplied {slice}
    auto tgt = slc.view(slice);

    // make a function that computes the magnitude of complex numbers
    auto magnitude = [] (slc_type::pixel_type pxl) -> cell_type
                     { return std::abs(pxl); };
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "pair #" << pid << ": target window start up: " << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // start the clock
    timer.reset().start();
    // initialize the tile by applying {magnitude} to every cell in {tgt} and storing the
    // result in {view}
    std::transform(tgt.begin(), tgt.end(), view.begin(), magnitude);
    // stop the clock
    timer.stop();
    // show me
    tlog
        << pyre::journal::at(__HERE__)
        << "reading target window, computing amplitudes, and storing: "
        << 1e6 * timer.read() << " μs"
        << pyre::journal::endl;

    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "adding target tile #" << pid << ":" << pyre::journal::newline
        << "    pair id: " << pid << pyre::journal::newline
        << "    slice:" << pyre::journal::newline
        << "        from: (" << slice.low() << ")" << pyre::journal::newline
        << "        to: (" << slice.high() << ")" << pyre::journal::newline
        << "    support:" << pyre::journal::newline
        << "        anchor: " << _hArena << pyre::journal::newline
        << "        offset: " << pid*(_refCells + _tgtCells) + _refCells << pyre::journal::newline
        << "        support: " << support << pyre::journal::newline
        << "        extent: " << tile.layout().size() << " cells" << pyre::journal::newline
        << "        footprint: " << sizeof(cell_type) * tile.layout().size() << " bytes"
        << pyre::journal::endl;

    // all done
    return;
}


void
ampcor::cuda::correlators::CUDA::
adjust(size_type wid)
{
    // move the input tiles to the device
    auto dArena = _newArena();
    // if something went wrong
    if (dArena == nullptr) {
        // nothing more to do
        return;
    }

    // build the SAT hyper-grid
    auto dSAT = _newSAT(dArena);
    // if something went wrong
    if (dSAT == nullptr) {
        // clean up
        cudaFree(dArena);
        // and bail
        return;
    }

    // use the SAT hyper-grid to pre-compute the amplitude averages for all possible target
    // placements of search windows in target tiles
    auto dAverage = _newAverageTargetAmplitudes(dSAT);
    // if something went wrong
    if (dAverage == nullptr) {
        // clean up
        cudaFree(dArena);
        cudaFree(dSAT);
        // and bail
        return;
    }

    // compute the correlation matrix
    auto dCorrelation = _newCorrelationMatrix(dArena, dAverage);
    // if something went wrong
    if (dCorrelation == nullptr) {
        // clean up
        cudaFree(dArena);
        cudaFree(dSAT);
        cudaFree(dAverage);
        // and bail
        return;
    }

    // clean up
    cudaFree(dArena);
    cudaFree(dSAT);
    cudaFree(dAverage);

    // all done
    return;
}


void
ampcor::cuda::correlators::CUDA::
refine(size_type wid)
{
    // all done
    return;
}


// meta-methods
ampcor::cuda::correlators::CUDA::
~CUDA() {
    delete [] _hArena;
}


ampcor::cuda::correlators::CUDA::
CUDA(size_type pairs, const shape_type & refShape, const shape_type & tgtShape) :
    _pairs{pairs},
    _refShape{refShape},
    _tgtShape{tgtShape},
    _corShape{_tgtShape - _refShape + index_type::fill(1)},
    _refCells(std::accumulate(_refShape.begin(), _refShape.end(), 1, std::multiplies<size_type>())),
    _tgtCells(std::accumulate(_tgtShape.begin(), _tgtShape.end(), 1, std::multiplies<size_type>())),
    _corCells(std::accumulate(_corShape.begin(), _corShape.end(), 1, std::multiplies<size_type>())),
    _refFootprint{_refCells * sizeof(cell_type)},
    _tgtFootprint{_tgtCells * sizeof(cell_type)},
    _corFootprint{_corCells * sizeof(cell_type)},
    _hArena{new cell_type [_pairs*(_refCells+_tgtCells)]}
{
    // compute the footprint
    auto footprint = _pairs*(_refCells + _tgtCells);

    // make a channel
    pyre::journal::debug_t channel("ampcor.cuda");
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "new CUDA worker:" << pyre::journal::newline
        << "    pairs: " << _pairs << pyre::journal::newline
        << "    ref shape: " << _refShape << pyre::journal::newline
        << "    tgt shape: " << _tgtShape << pyre::journal::newline
        << "    footprint: " << footprint << " cells in " << (8.0*footprint/1024/1024) << " Mb"
        << pyre::journal::endl;
}


// implementation details
auto
ampcor::cuda::correlators::CUDA::
_newArena() const -> cell_type *
{
    // make a timer
    timer_t timer("ampcor.cuda");
    // make a channel
    pyre::journal::info_t channel("ampcor.cuda");
    // and another for reporting timings
    pyre::journal::info_t tlog("ampcor.cuda.timings");

    // compute the total amount of memory required for all the cells in the input hyper-grid
    auto footprint = _pairs * (_refFootprint + _tgtFootprint);
    // pick a spot
    cell_type * dArena = nullptr;
    // allocate device memory
    cudaError_t status = cudaMallocManaged(&dArena, footprint);
    // if something went wrong
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while allocating device memory for the input hyper-grid: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // bail
        return nullptr;
    }
    // show me
    channel
        << pyre::journal::at(__HERE__)
        << "allocated an arena of " << (footprint) << " bytes for the input hyper-grid at "
        << dArena
        << pyre::journal::endl;

    // transfer the data to the device
    // start the clock
    timer.reset().start();
    // transfer the data
    status = cudaMemcpy(dArena, _hArena, footprint, cudaMemcpyHostToDevice);
    // stop the clock
    timer.stop();
    // check
    if (status != cudaSuccess) {
        // make a channel
        pyre::journal::error_t channel("ampcor.cuda");
        // complain
        channel
            << pyre::journal::at(__HERE__)
            << "while transferring the tiles to the device: "
            << cudaGetErrorName(status) << " (" << status << ")"
            << pyre::journal::endl;
        // release device memory
        cudaFree(dArena);
        // and bail
        return nullptr;
    }
    // read the timer
    auto elapsed = timer.read();
    // compute the bandwidth
    auto bandwidth = footprint / elapsed;
    // report the timing
    tlog
        << pyre::journal::at(__HERE__)
        << "moving reference and target data to the device: " << 1e6 * elapsed << " μs"
        << " at " << (bandwidth/1024/1024/1024) << " Gb/s"
        << pyre::journal::endl;

    // all done
    return dArena;
}


// end of file
