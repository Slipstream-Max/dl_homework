from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "dl_utils.im2col_cython",
        ["dl_utils/im2col_cython.pyx"],
        include_dirs=[numpy.get_include()],
    ),
]

setup(
    name="dl_utils",
    version="0.1",
    packages=find_packages(),
    package_data={"dl_utils": ["datasets/cifar-10-batches-py/*"]},
    include_package_data=True,
    ext_modules=cythonize(extensions),
    install_requires=["numpy", "cython"],
)
