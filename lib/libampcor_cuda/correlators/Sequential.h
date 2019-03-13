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
    // the underlying pixel complex type
    using cell_type = typename raster_type::cell_type;
    // the support of the pixel complex type
    using value_type = typename cell_type::value_type;
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

    // coarse adjustment to the offset map
    void adjust();
    // fine adjustments to the offset map
    void refine();

    // accessors
    auto arena() const -> const cell_type *;

    // debugging support
    void dump() const;

    // meta-methods
public:
    virtual ~Sequential();
    Sequential(size_type pairs, const layout_type & refLayout, const layout_type & tgtLayout);


    // implementation details: methods
public:
    // push the tiles in the plan to device
    auto _push() const -> cell_type *;
    // compute the magnitude of the complex signal pixel-by-pixel
    auto _detect(const cell_type * cArena) const -> value_type *;
    // subtract the mean from reference tiles and compute the square root of their variance
    auto _refStats(value_type * rArena) const -> value_type *;
    // compute the sum area tables for the target tiles
    auto _sat(const value_type * rArena) const -> value_type *;
    // compute the mean of all possible placements of a tile the same size as the reference
    // tile within the target
    auto _tgtStats(const value_type * sat) const -> value_type *;
    // correlate
    auto _correlate(const value_type * rArena,
                    const value_type * refStats,
                    const value_type * tgtStats) const -> value_type *;

    // implementation details: data
private:
    // my capacity, in {ref/tgt} pairs
    size_type _pairs;

    // the shape of the reference tiles
    layout_type _refLayout;
    // the shape of the search windows in the target image
    layout_type _tgtLayout;
    // the shape of the correlation matrix
    layout_type _corLayout;

    // the number of cells in a reference tile
    size_type _refCells;
    // the number of cells in a target search window
    size_type _tgtCells;
    // the number of cell in a correlation matrix
    size_type _corCells;

    // the number of bytes in a reference tile
    size_type _refFootprint;
    // the number of bytes in a target search window
    size_type _tgtFootprint;
    // the number of bytes in a correlation matrix
    size_type _corFootprint;

    // host storage for the tile pairs
    cell_type * _arena;
};


// code guard
#endif

// end of file
