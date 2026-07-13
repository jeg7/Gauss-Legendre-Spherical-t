# Gauss-Legendre-Spherical-*t* (GLST)

## About
GLST is a standalone CUDA/C++ library that implements the
Gauss-Legendre-Spherical-*t* algorithm for electrostatic interactions.

If you use GLST in published work, please cite the following publications:

[W. Hwang, J. E. Gonzales II, and B. R. Brooks, "Gauss-Legendre-Spherical-*t*
(GLST) cubature-based factorization of long-range electrostatics in
simulations," J. Chem. Phys. 162, 224102
(2025).](https://pubs.aip.org/aip/jcp/article/162/22/224102/3349221)

[J. E. Gonzales II, W. Hwang, and B. R. Brooks, "A parallel CUDA implementation
of the Gauss-Legendre-Spherical-*t* method for electrostatic interactions," J.
Chem. Phys. 162, 222501
(2025).](https://pubs.aip.org/aip/jcp/article/162/22/222501/3349220)

## License

GLST is distributed under the
[BSD 3-Clause](https://opensource.org/licenses/BSD-3-Clause) open source
license, as described in the `LICENSE` file in the top-level of the repository.

## Dependencies

GLST requires the NVIDIA Collective Communications Library (NCCL) to be
installed on the system. For the GLST build process to find the installation of
NCCL, you can define the environment variable `NCCL_ROOT` as the path where
NCCL is installed. For example, if NCCL has been installed in
`/home/user/software/nccl/build`, you should add
`export NCCL_ROOT=/home/user/software/nccl/build` in your shell's configuration
file (e.g. ~/.bashrc).

## Authors

James E. Gonzales II (NIH)

Wonmuk Hwang (Texas A&M University)

Bernard R. Brooks (NIH)

## Installation

The source code was developed using the following tool and compiler versions.
Other versions may work.

* GCC [12.2.0]
* CUDA [12.2.140]
* CMake [3.25.1]
* NCCL [2.29.2-1]

### 0. Clone this repository

In the directory you would like to install GLST, run:
```
git clone https://github.com/jeg7/Gauss-Legendre-Spherical-t.git
cd Gauss-Legendre-Spherical-t/
```

### 1. Compile the source code

For a standard release build:
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

## Getting Help and Contributing

Please contact <james.gonzales@nih.gov> for any of the following:

* Installation issues
* Usage questions
* Bug reports
* Contribution proposals
