# CircuitSeeker
---

Python tools for multimodal integration of microscopy data.

* Highly configurable registration of multimodal image data.
* Distributed motion correction of large 4D datasets.
* Composition and inversion of affine and vector field transforms.
* Distributed deltaF over F calculation.

|                          |                          |
:-------------------------:|:-------------------------:
![cross_fade](resources/cross_fade_two_axis_downsample.gif)  |  ![alignment](resources/exR0_to_conf_registration.gif)


CircuitSeeker contains rigid, affine, deformable, and highly customizable piecewise/overlapping-blockwise registration algorithms. The modular design supports quick construction of simple to highly complex alignment pipelines quickly. This is important because every microscopy dataset is unique. CircuitSeeker strives to be both simple and flexible.

## Installation
`pip install CircuitSeeker`

## Versions
* Current dev (on github): 0.4.2
* Current stable (on PyPI): 0.4.1


I don't have a proper system for versioning. I bump minor version when I add small features, mid version when a refactor affects the API, and have yet to bump the major version number. I try to keep the github version more current than PyPI, which is hopefully a stable implementation.

## Documentation
Most user level functions have docstrings, which I strive to keep up to date, but much more work is needed. Don't hesitate to reach out with questions. **Please use the github issue tracker for questions and support.**

## Modules
`CircuitSeeker.align`
* random affine search
* rigid and affine
* deformable
* prebuilt pipelines
* distributed pipelines


`CircuitSeeker.transform`
* Apply, compose, and invert transforms


`CircuitSeeker.utility`
* Conversions between different image and transform formats


`CircuitSeeker.motion_correct`
* 4D motion correction pipelines
* rigid, affine, and (if you really need it) fast deformable

`CircuitSeeker.level_set`
* level set segmentation for foreground detection

`CircuitSeeker.function`
* distributed deltaF over F

## Examples
I've included some of my own [Jupyter notebooks here](https://github.com/GFleishman/CircuitSeeker/tree/master/notebooks). These were not designed as tutorials and bear in mind that some aspects (especially anything distributed or requiring a cluster) are specific to my own computing environment. But these notebooks will still be helpful in getting you started. Much more work is needed to properly describe/share the full range of CircuitSeeker capabilities.

## Dependencies
These amazing packages make CircuitSeeker possible. They are automatically installed with CircuitSeeker:
* [SimpleITK](https://github.com/SimpleITK/SimpleITK)
* [Dask](https://github.com/dask/dask)
* [ClusterWrap](https://github.com/GFleishman/ClusterWrap)
* [numpy](https://github.com/numpy/numpy)
* [scipy](https://github.com/scipy/scipy)
* [h5py](https://github.com/h5py/h5py)
* [zarr](https://github.com/zarr-developers/zarr-python)
* [morphsnakes](https://github.com/pmneila/morphsnakes)
* [pyfftw](https://github.com/pyFFTW/pyFFTW)
