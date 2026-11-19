"""Load the trained LIBSVM probability model through its native C API."""
import ctypes as c
from ctypes.util import find_library
import os
import sys
from weakref import finalize


class Node(c.Structure):
    _fields_ = [('index', c.c_int), ('value', c.c_double)]


class ProbabilityModel:
    def __init__(self, path, classes):
        library = find_library('svm')
        if library is None and sys.platform == 'darwin':
            library = '/usr/lib/libCRFSuite.dylib'
        if not library:
            raise RuntimeError('LIBSVM runtime unavailable')
        lib = c.CDLL(library)
        lib.svm_load_model.argtypes = [c.c_char_p]
        lib.svm_load_model.restype = c.c_void_p
        lib.svm_free_and_destroy_model.argtypes = [c.POINTER(c.c_void_p)]
        lib.svm_free_and_destroy_model.restype = None
        lib.svm_get_nr_class.argtypes = [c.c_void_p]
        lib.svm_check_probability_model.argtypes = [c.c_void_p]
        lib.svm_get_labels.argtypes = [c.c_void_p, c.POINTER(c.c_int)]
        lib.svm_get_labels.restype = None
        lib.svm_predict_probability.argtypes = [c.c_void_p, c.POINTER(Node), c.POINTER(c.c_double)]
        lib.svm_predict_probability.restype = c.c_double
        self.lib = lib
        self.handle = c.c_void_p(lib.svm_load_model(os.fsencode(path)))
        if not self.handle:
            raise ValueError('Invalid LIBSVM model')
        finalize(self, lib.svm_free_and_destroy_model, c.pointer(self.handle))
        if lib.svm_get_nr_class(self.handle) != len(classes) or not lib.svm_check_probability_model(self.handle):
            raise ValueError('Unexpected LIBSVM probability or class metadata')
        labels = (c.c_int * len(classes))()
        lib.svm_get_labels(self.handle, labels)
        if sorted(labels) != list(range(len(classes))):
            raise ValueError('Unknown class IDs')
        self.classes = classes
        self.probability_order = [classes[i] for i in labels]

    def predict(self, scaled):
        nodes = (Node * (len(scaled) + 1))()
        for index, value in enumerate(scaled, start=1):
            nodes[index - 1] = Node(index, value)
        nodes[-1] = Node(-1, 0)
        probabilities = (c.c_double * len(self.classes))()
        label = int(self.lib.svm_predict_probability(self.handle, nodes, probabilities))
        return self.classes[label], dict(zip(self.probability_order, probabilities))
