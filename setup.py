from setuptools import setup, find_packages, Extension
import numpy

# # Try to include Cython extension if available
# try:
#     from Cython.Build import cythonize
#     extensions = [
#         Extension(
#             "dl_utils.im2col_cython", ["dl_utils/im2col_cython.pyx"],
#             include_dirs=[numpy.get_include()]
#         ),
#     ]
#     ext_modules = cythonize(extensions)
# except ImportError:
#     ext_modules = []

setup(
    name="dl_utils",
    version="0.1.0",
    description="Deep Learning Utilities",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'dl_utils': ['datasets/*'],
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'imageio',
        'six',
    ],
    # ext_modules=ext_modules,
)
