import setuptools

setuptools.setup(
    name="CircuitSeeker",
    version="0.0.1",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for finding neural circuits",
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
