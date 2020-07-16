import setuptools

setuptools.setup(
    name="CircuitSeeker",
    version="0.0.1",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for finding neural circuits",
    url="https://github.com/GFleishman/CircuitSeeker",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'dask[complete]',
        'dask-jobqueue',
        'SimpleITK',
        'zarr',
        'numcodecs',
    ]
)
