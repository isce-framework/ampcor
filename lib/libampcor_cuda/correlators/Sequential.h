// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_Sequential_h)
#define ampcor_libampcor_cuda_correlators_Sequential_h


// resource management and orchestration of the execution of the correlation plan
template <typename raster_t>
class ampcor::cuda::correlators::Sequential {
    // types
public:
    // my client raster type
    using raster_type = raster_t;
    // views over it
    using view_type = typename raster_type::view_type;
    using constview_type = typename raster_type::constview_type;
    // the underlying pixel type
    using cell_type = typename raster_type::cell_type;
    // for describing slices of rasters
    using slice_type = typename raster_type::slice_type;
    // for describing the shapes of tiles
    using shape_type = typename raster_type::shape_type;
    // for describing the layouts of tiles
    using layout_type = typename raster_type::layout_type;
    // for index arithmetic
    using index_type = typename raster_type::index_type;
    // for sizing things
    using size_type = typename raster_type::size_type;

    // adapter for tiles within my arena
    using tile_type = pyre::grid::grid_t<cell_type,
                                         layout_type,
                                         pyre::memory::view_t<cell_type>>;

    // interface
public:
    // add a reference tile to the pile
    void addReferenceTile(size_type pid, const constview_type & ref);
    // add a target search window to the pile
    void addTargetTile(size_type pid, const constview_type & tgt);

    // debugging support
    void dump() const;

    // meta-methods
public:
    virtual ~Sequential();
    Sequential(size_type pairs, const layout_type & refLayout, const layout_type & tgtLayout);

    // implementation details: data
private:
    // my capacity, in {ref/tgt} pairs
    size_type _pairs;

    // the shape of the reference tiles
    layout_type _refLayout;
    // the shape of the search windows in the target image
    layout_type _tgtLayout;

    // the number of cells in a reference tile
    size_type _refCells;
    // the number of cells in a target search window
    size_type _tgtCells;

    // the number of bytes in a reference tile
    size_type _refFootprint;
    // the number of bytes in a target search window
    size_type _tgtFootprint;

    // host storage for the tile pairs
    cell_type * _hArena;
};


// code guard
#endif

// end of file