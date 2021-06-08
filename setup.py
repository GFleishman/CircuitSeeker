import setuptools

setuptools.setup(
    name="CircuitSeeker",
    version="0.2.2",
    author="Greg M. Fleishman",
    author_email="greg.nli10me@gmail.com",
    description="Tools for finding neural circuits",
    url="https://github.com/GFleishman/CircuitSeeker",
    license="MIT",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'h5py',
        'bokeh',
        'dask',
        'dask[array]',
        'dask[bag]',
        'dask[delayed]',
        'dask[distributed]',
        'dask-jobqueue',
        'SimpleITK',
        'zarr',
        'numcodecs',
        'morphsnakes',
        'ClusterWrap>=0.1.6',
        'pynrrd',
        'greedypy',
        'dask-stitch>=0.2.1',
    ]
)
