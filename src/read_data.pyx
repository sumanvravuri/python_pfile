from __future__ import division
import numpy as np
cimport numpy as np
cimport cython
from libc.stdio cimport *

DTYPEINT = np.int32
ctypedef np.int32_t DTYPEINT_t

DTYPEFLOAT = np.float32
ctypedef np.float32_t DTYPEFLOAT_t

cdef extern from "stdio.h":
    FILE *fdopen(int, const char *)

def swap_float(float x):
    pass
    
def read_feature_data(int fid_no, np.ndarray[DTYPEFLOAT_t, ndim=2] data, int header_bytes_skip, int num_labels_skip):
    cdef int frame_index, dim_index
    cdef int num_frames_to_read = data.shape[0]
    cdef int num_dims = data.shape[1]
#    print num_dims
    cdef FILE* cfile
#    cdef np.ndarray[DTYPE_t, ndim=1] data
    cdef DTYPEFLOAT_t* data_ptr

    cfile = fdopen(fid_no, 'rb') # attach the stream
#    data = np.zeros(count).astype(DTYPE)
    data_ptr = <DTYPEFLOAT_t*>data.data
    
    #skip header
    fseek(cfile, header_bytes_skip, SEEK_SET)

    for frame_index in range(num_frames_to_read):
        if fseek(cfile, 2 * sizeof(DTYPEINT_t), SEEK_CUR): #don't read frame_info
            break
        dim_index = fread(<void*>(data_ptr + frame_index * num_dims), sizeof(DTYPEFLOAT_t), num_dims, cfile)
        if dim_index < num_dims:
#            print "breaking with", dim_index, "at frame_index", frame_index, "of", num_frames_to_read, sizeof(DTYPEFLOAT_t)
            break
        if fseek(cfile, num_labels_skip * sizeof(DTYPEINT_t), SEEK_CUR):
            break

#    return data