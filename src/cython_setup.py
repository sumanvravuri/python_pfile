'''
Created on Jan 2, 2014

@author: sumanravuri
'''

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup (
    cmdclass = {'build_ext' : build_ext},
    ext_modules = [
        Extension("read_data",["read_data.pyx"],
                  include_dirs=[numpy.get_include()]
        ),
])