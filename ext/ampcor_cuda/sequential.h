// -*- C++ -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

#if !defined(ampcor_extension_sequential_h)
#define ampcor_extension_sequential_h


// place everything in my private namespace
namespace ampcor {
    namespace extension {
        namespace cuda {
            namespace sequential {

                // instantiate a new sequential
                extern const char * const alloc__name__;
                extern const char * const alloc__doc__;
                PyObject * alloc(PyObject *self, PyObject *args);

                // compute the amplitude of a reference chip and save it in my dataspace
                extern const char * const addReference__name__;
                extern const char * const addReference__doc__;
                PyObject * addReference(PyObject *self, PyObject *args);

                // compute the amplitude of a target chip and save it in my dataspace
                extern const char * const addTarget__name__;
                extern const char * const addTarget__doc__;
                PyObject * addTarget(PyObject *self, PyObject *args);

                // perform pixel level adjustments to the registration map
                extern const char * const adjust__name__;
                extern const char * const adjust__doc__;
                PyObject * adjust(PyObject *self, PyObject *args);

                // perform sub-pixel level adjustments to the registration map
                extern const char * const refine__name__;
                extern const char * const refine__doc__;
                PyObject * refine(PyObject *self, PyObject *args);

            } // of namespace sequential`
        } // of namespace cuda
    } // of namespace extension`
} // of namespace ampcor

#endif

// end of file
