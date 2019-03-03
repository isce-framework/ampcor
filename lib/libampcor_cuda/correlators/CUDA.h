// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_CUDA_h)
#define ampcor_libampcor_cuda_correlators_CUDA_h


// access to the dom
#include <ampcor/dom.h>

// resource management and orchestration of the execution of the correlation plan
class ampcor::cuda::correlators::CUDA {
    // types
public:
    // my storage type
    using cell_type = double;
    // my client raster type
    using slc_type = ampcor::dom::slc_t;
    // for describing slices of rasters
    using slice_type = slc_type::slice_type;
    // for describing the shapes of tiles
    using shape_type = slc_type::shape_type;
    // for index arithmetic
    using index_type = slc_type::index_type;
    // for sizing things
    using size_type = slc_type::size_type;

    // i use {cell_type} grids that ride on top of my dataspace with the same layout as the SLC
    using gview_type = pyre::grid::grid_t<cell_type,
                                          slc_type::layout_type,
                                          pyre::memory::view_t<cell_type>>;

    // interface
public:
    // add a reference tile to the pile
    void addReferenceTile(size_type pid, const slc_type & slc, slice_type slice);
    // add a target search window to the pile
    void addTargetTile(size_type pid, const slc_type & slc, slice_type slice);

    // perform pixel level adjustments to the registration map
    void adjust(size_type wid = 0);
    // perform sub-pixel level adjustments to the registration map
    void refine(size_type wid = 0);

    // meta-methods
public:
    virtual ~CUDA();
    CUDA(size_type pairs, const shape_type & refShape, const shape_type & tgtShape);

    // implementation details: methods
private:
    auto _newArena() const -> cell_type *;
    auto _newSAT(const cell_type * dArena) const -> cell_type *;
    auto _newAverageTargetAmplitudes(const cell_type * dSAT) const -> cell_type *;
    auto _newCorrelationMatrix(const cell_type * dArena,
                               const cell_type * dAverage) const -> cell_type *;

    // implementation details: data
private:
    // my capacity, in {ref/tgt} pairs
    size_type _pairs;

    // the shape of the reference tiles
    shape_type _refShape;
    // the shape of the search windows in the target image
    shape_type _tgtShape;
    // the shape of the correlation matrix
    shape_type _corShape;

    // the number of cells in a reference tile
    size_type _refCells;
    // the number of cells in a target search window
    size_type _tgtCells;
    // the number of cells in a correlation matrix
    size_type _corCells;

    // the number of bytes in a reference tile
    size_type _refFootprint;
    // the number of bytes in a target search window
    size_type _tgtFootprint;
    // the number of bytes in a correlation matrix
    size_type _corFootprint;

    // host storage for the tile pairs
    cell_type * _hArena;
};


// code guard
#endif

// end of file
