// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// configuration
#include <portinfo>

// externals
#include <Python.h>
#include <pyre/journal.h>
// support
#include <ampcor_cuda/correlators.h>
// access to the {ampcor} extension capsules
#include <ampcor/capsules.h>

// my declarations
#include "capsules.h"
#include "sequential.h"


// alias SLC
using slc_t = ampcor::dom::slc_t;
// alias the sequential worker
using sequential_t = ampcor::cuda::correlators::sequential_t<slc_t>;


// constructor
const char * const
ampcor::extension::cuda::sequential::
alloc__name__ = "sequential";

const char * const
ampcor::extension::cuda::sequential::
alloc__doc__ = "instantiate a new sequential correlation worker";

PyObject *
ampcor::extension::cuda::sequential::
alloc(PyObject *, PyObject *args)
{
    // the number of pairs
    std::size_t pairs;
    std::size_t refLines, refSamples;
    std::size_t tgtLines, tgtSamples;
    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "k(kk)(kk):sequential",
                              &pairs,
                              &refLines, &refSamples, &tgtLines, &tgtSamples);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // build the reference tile shape
    sequential_t::shape_type ref { refSamples, refLines };
    // and the target tile shape
    sequential_t::shape_type tgt { tgtSamples, tgtLines };


    // instantiate the worker
    sequential_t * worker = new sequential_t(pairs, ref, tgt);
    // dress it up and return it
    return PyCapsule_New(worker, capsule_t, free);
}


// read a reference tile and save it in the dataspace
const char * const
ampcor::extension::cuda::sequential::
addReference__name__ = "addReference";

const char * const
ampcor::extension::cuda::sequential::
addReference__doc__ = "process a reference tile";

PyObject *
ampcor::extension::cuda::sequential::
addReference(PyObject *, PyObject *args)
{
    PyObject * pyWorker;
    PyObject * pySLC;
    std::size_t idx;
    std::size_t beginLine, beginSample;
    std::size_t endLine, endSample;
    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!O!k(kk)(kk):addReference",
                              &PyCapsule_Type, &pyWorker,
                              &PyCapsule_Type, &pySLC,
                              &idx,
                              &beginLine, &beginSample, &endLine, &endSample);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the worker capsule
    if (!PyCapsule_IsValid(pyWorker, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid worker capsule");
        // and bail
        return nullptr;
    }
    // and unpack it; can't be const
    sequential_t & worker =
        *reinterpret_cast<sequential_t *>(PyCapsule_GetPointer(pyWorker, capsule_t));

    // make an alias for the SLC capsule
    const char * const slcCapsule_t = ampcor::extension::slc::capsule_t;
    // check the SLC capsule
    if (!PyCapsule_IsValid(pySLC, slcCapsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid SLC capsule");
        // and bail
        return nullptr;
    }
    // and unpack it
    const slc_t & slc =
        *reinterpret_cast<const slc_t *>(PyCapsule_GetPointer(pySLC, slcCapsule_t));

    // build a description of the slice for this tile
    slc_t::index_type begin { beginLine, beginSample };
    slc_t::index_type end { endLine, endSample };
    // convert it into a slice
    auto slice = slc.layout().slice(begin, end);

    // ask the worker to add to its pile the reference tile described by {slice}
    worker.addReferenceTile(idx, slc.constview(slice));

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}


// read a target tile and save it in the dataspace
const char * const
ampcor::extension::cuda::sequential::
addTarget__name__ = "addTarget";

const char * const
ampcor::extension::cuda::sequential::
addTarget__doc__ = "process a target tile";

PyObject *
ampcor::extension::cuda::sequential::
addTarget(PyObject *, PyObject *args)
{
    PyObject * pyWorker;
    PyObject * pySLC;
    std::size_t idx;
    std::size_t beginLine, beginSample;
    std::size_t endLine, endSample;
    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!O!k(kk)(kk):addTarget",
                              &PyCapsule_Type, &pyWorker,
                              &PyCapsule_Type, &pySLC,
                              &idx,
                              &beginLine, &beginSample, &endLine, &endSample);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the worker capsule
    if (!PyCapsule_IsValid(pyWorker, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid Sequential worker capsule");
        // and bail
        return nullptr;
    }
    // and unpack it; can't be const
    sequential_t & worker =
        *reinterpret_cast<sequential_t *>(PyCapsule_GetPointer(pyWorker, capsule_t));

    // make an alias for the SLC capsule
    const char * const slcCapsule_t = ampcor::extension::slc::capsule_t;
    // check the SLC capsule
    if (!PyCapsule_IsValid(pySLC, slcCapsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid SLC capsule");
        // and bail
        return nullptr;
    }
    // and unpack it
    const slc_t & slc =
        *reinterpret_cast<const slc_t *>(PyCapsule_GetPointer(pySLC, slcCapsule_t));

    // build a description of the slice for this tile
    slc_t::index_type begin { beginLine, beginSample };
    slc_t::index_type end { endLine, endSample };
    // convert it into a slice
    auto slice = slc.layout().slice(begin, end);

    // ask the worker to add to its pile the target tile described by {slice}
    worker.addTargetTile(idx, slc.constview(slice));

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}


// perform pixel level adjustments to the registration map
const char * const
ampcor::extension::cuda::sequential::
adjust__name__ = "adjust";

const char * const
ampcor::extension::cuda::sequential::
adjust__doc__ = "perform pixel adjustments to the registration map";

PyObject *
ampcor::extension::cuda::sequential::
adjust(PyObject *, PyObject *args)
{
    PyObject * pyWorker;
    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!:adjust",
                              &PyCapsule_Type, &pyWorker);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the worker capsule
    if (!PyCapsule_IsValid(pyWorker, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid Sequential worker capsule");
        // and bail
        return nullptr;
    }
    // and unpack it; can't be const
    sequential_t & worker =
        *reinterpret_cast<sequential_t *>(PyCapsule_GetPointer(pyWorker, capsule_t));

    // ask the worker to perform pixel level adjustments to the registration map
    worker.adjust();

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// perform sub-pixel level adjustments to the registration map
const char * const
ampcor::extension::cuda::sequential::
refine__name__ = "refine";

const char * const
ampcor::extension::cuda::sequential::
refine__doc__ = "perform sub-pixel adjustments to the registration map";

PyObject *
ampcor::extension::cuda::sequential::
refine(PyObject *, PyObject *args)
{
    PyObject * pyWorker;
    // attempt to parse the arguments
    int ok = PyArg_ParseTuple(args,
                              "O!:refine",
                              &PyCapsule_Type, &pyWorker);
    // if something went wrong
    if (!ok) {
        // complain
        return nullptr;
    }

    // check the worker capsule
    if (!PyCapsule_IsValid(pyWorker, capsule_t)) {
        // give a reason
        PyErr_SetString(PyExc_TypeError, "invalid Sequential worker capsule");
        // and bail
        return nullptr;
    }
    // and unpack it; can't be const
    sequential_t & worker =
        *reinterpret_cast<sequential_t *>(PyCapsule_GetPointer(pyWorker, capsule_t));

    // ask the worker to refine the registration map
    worker.refine();

    // all done
    Py_INCREF(Py_None);
    return Py_None;
}

// destructors
void
ampcor::extension::cuda::sequential::
free(PyObject * capsule) {
    // bail out if the capsule is not valid
    if (!PyCapsule_IsValid(capsule, capsule_t)) return;
    // get the matrix
    sequential_t * worker =
        reinterpret_cast<sequential_t *>(PyCapsule_GetPointer(capsule, capsule_t));
    // deallocate
    delete worker;
    // and return
    return;
}




// end of file
