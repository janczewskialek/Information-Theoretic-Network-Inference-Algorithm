from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy
import sys
from os.path import join


if sys.platform == "win32":
    flag_speed = '/O2'
    cpp_version = '/std:c++17'
elif sys.platform in ["linux", "darwin"]:
    flag_speed = '-O3'
    cpp_version = '-std=c++17'
else:
    raise('Specify the proper compilation flag to optimize the code for speed on', sys.platform, '!')


ckdtree_src = ["query.cxx",
               "build.cxx",
               "query_pairs.cxx",
               "count_neighbors.cxx",
               "query_ball_point.cxx",
               "query_ball_tree.cxx",
               "sparse_distances.cxx"]

ckdtree_headers = ["ckdtree_decl.h",
                   "coo_entries.h",
                   "distance_base.h",
                   "distance.h",
                   "ordered_pair.h",
                   "rectangle.h"]

ckdtree_src = [join("../../../TransferEntropy/ckdtree", "src", x) for x in ckdtree_src]
ckdtree_headers = [join("../../../TransferEntropy/ckdtree", "src", x) for x in ckdtree_headers]
ckdtree_headers += "../../../TransferEntropy/circ_shift.h"
ckdtree_dep = ckdtree_src+ckdtree_headers

sources = ['code_cython/fastvecm.pyx']
sources.extend(ckdtree_src)
sources.append("../../../TransferEntropy/INA.cpp")

extensions = Extension('fastvecm',
                       sources= sources,
                       extra_compile_args=[flag_speed, cpp_version],
                       depends=ckdtree_dep + ["../../../TransferEntropy/INA.h"],
                       language='c++')

setup(
    name='VECM package',
    ext_modules=cythonize(extensions,
                          compiler_directives={'language_level': "3"}
                          ),
    include_dirs=[numpy.get_include(), "code_cpp", "code_cython", "../../../TransferEntropy/ckdtree/src/", "../../../TransferEntropy"],
    cmdclass={'build_ext': build_ext}
)
