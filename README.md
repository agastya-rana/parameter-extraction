# FRET-data-assimilation

Estimate parameters and hidden states of dynamical models from single-cell FRET data using *variational annealing* (Ye et al, *Phys. Rev. E* **92**, 052901, 2015), a technique of nonlinear data assimilation for pinpointing global minima among highly non-convex cost functions.

Effective numerical optimization of cost functions requires their first (and possibly, second) derivatives. In practice, coding these derivatives can be prohibitive for complicated dynamical models. This code utilizes a NumPy-based implementation of [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation), which precludes the need to manually code these derivatives.



## Getting Started

### Prerequisites

FRET-data-assimilation itself is standalone. To run the scripts, the following must be installed on your computer: 

1. Python 2 (tested on version 2.7.13)
2. SciPy (tested on version 0.17.0)
3. NumPy (tested on version 1.10.4 )
4. [VarAnneal](https://github.com/paulrozdeba/varanneal), a Python implementation of variational annealing that utilizes automatic differentiation
5. [PYADOLC](https://github.com/b45ch1/pyadolc), a Python implementation of automatic differentiation

To install items 4 and 5, follow the instructions in the VarAnneal repository readme. 

### Tests: todo



## Usage

Generally speaking, the data assimilation procedure for FRET data consists of 3 steps:

1. Record data or generate synthetic (fake) data
2. Generate many estimates of unknowns using data assimilation with a subset of recorded data
3. Find optimal parameter estimates by comparing predictions from each estimated variable set against remainder of data

The first 



## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc
