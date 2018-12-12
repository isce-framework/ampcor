// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2018 all rights reserved
//

// configuration
#include <portinfo>

// externals
#include <Python.h>
#include <pyre/journal.h>
// support
#include <ampcor/dom.h>

// my declarations
#include "slc.h"
#include "capsules.h"


// compute the pixel size of SLC rasters
// map
const char * const
ampcor::extension::slc::
pixelSize__name__ = "slc_pixelSize";

const char * const
ampcor::extension::slc::
pixelSize__doc__ = "the size of individual pixels in SLC raster images";

PyObject *
ampcor::extension::slc::
pixelSize(PyObject *, PyObject *)
{
    // easy enough
    return PyLong_FromLong(ampcor::dom::slc_t::pixelSize());
}


// map
const char * const
ampcor::extension::slc::
map__name__ = "slc_map";

const char * const
ampcor::extension::slc::
map__doc__ = "memory map an SLC raster of a given shape";

PyObject *
ampcor::extension::slc::
map(PyObject *, PyObject *args, PyObject *kwds)
{
    // storage for the file name
    const char * filename;
    // storage for the raster shape
    size_t samples=0, lines=0;
    // the list argument name
    static const char * names[] = {"filename", "samples", "lines", nullptr};
    // attempt to parse the arguments
    int ok = PyArg_ParseTupleAndKeywords(args,
                                         kwds,
                                         "skk:slc_map",
                                         const_cast<char**>(names),
                                         &filename, &samples, &lines);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // instantiate the image
    ampcor::dom::slc_t * slc = new ampcor::dom::slc_t(filename, {lines, samples});

    // make a channel
    pyre::journal::debug_t channel("ampcor.slc");
    // show me what we have so for
    channel
        << pyre::journal::at(__HERE__)
        << "slc_map:"
        << pyre::journal::newline
        << "    filename: '" << filename << "'"
        << pyre::journal::newline
        << "    shape: (" << lines << " x " << samples << ")"
        << pyre::journal::newline
        << "    size: " << slc->size() << " bytes at " << slc->data()
        << pyre::journal::newline
        << "    data[0,0]: " << (*slc)[{0u,0u}]
        << pyre::journal::endl;

    // dress it up and return it
    return PyCapsule_New(slc, capsule_t, free);
}


// fetch data at the given index
const char * const
ampcor::extension::slc::
atIndex__name__ = "slc_atIndex";

const char * const
ampcor::extension::slc::
atIndex__doc__ = "fetch the data at the given index";

PyObject *
ampcor::extension::slc::
atIndex(PyObject *, PyObject *args)
{
    // storage for the indices
    size_t i, j;
    // storage for the capsule
    PyObject * capsule;

    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!kk:slc_atIndex",
                              &PyCapsule_Type, &capsule, &i, &j);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the capsule
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid SLC capsule");
        // and bail
        return nullptr;
    }

    // unpack the capsule
    const ampcor::dom::slc_t & slc =
        *reinterpret_cast<const ampcor::dom::slc_t *>(PyCapsule_GetPointer(capsule, capsule_t));
    // get the data
    auto value = slc[{i, j}];

    // dress it up and return it
    return PyComplex_FromDoubles(value.real(), value.imag());
}


// fetch data at the given offset
const char * const
ampcor::extension::slc::
atOffset__name__ = "slc_atOffset";

const char * const
ampcor::extension::slc::
atOffset__doc__ = "fetch the data at the given offset";

PyObject *
ampcor::extension::slc::
atOffset(PyObject *, PyObject *args)
{
    // storage for the offset
    size_t offset;
    // storage for the capsule
    PyObject * capsule;

    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!k:slc_atOffset",
                              &PyCapsule_Type, &capsule, &offset);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the capsule
    if (!PyCapsule_IsValid(capsule, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid SLC capsule");
        // and bail
        return nullptr;
    }

    // unpack the capsule
    const ampcor::dom::slc_t & slc =
        *reinterpret_cast<const ampcor::dom::slc_t *>(PyCapsule_GetPointer(capsule, capsule_t));
    // get the data
    auto value = slc[offset];

    // dress it up and return it
    return PyComplex_FromDoubles(value.real(), value.imag());
}


// destructors
void
ampcor::extension::slc::
free(PyObject * capsule) {
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the matrix
    ampcor::dom::slc_t * slc =
        reinterpret_cast<ampcor::dom::slc_t *>(PyCapsule_GetPointer(capsule, capsule_t));
    // deallocate
    delete slc;
    // and return
    return;
}




// end of file
