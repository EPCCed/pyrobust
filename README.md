Robust optimisation aims to find solutions that, given a particular measure of robustness, remain optimal in the face of perturbations within a defined disturbance neighbourhood. pyrobust is a python implementation of various robust optimisation methods developed by the University of Exeter (Prof. Jonathan E. Fieldsend's group). The core optimisation methods utilise the population's search history and resamples the best individuals for new solutions, as described in: _[On the Exploitation of Search History and Accumulative Sampling in Robust Optimisation](https://ore.exeter.ac.uk/repository/handle/10871/27155)_ by K. Alyahya, K. Doherty, O. E. Akman and J. E. Fieldsend, GECCO ’17 Companion, July 15-19, 2017.

Please reference this paper if you undertake work utilising this code.

Documentation for pyrobust can be found at: https://github.com/EPCCed/pyrobust/wiki

## Install pyrobust

The Python implementation of Robust requires Python 3, Numpy (https://www.numpy.org/), Scipy (https://www.scipy.org/), pyDOE for Latin Hypercube sampling (https://pythonhosted.org/pyDOE/) and sobol_seq for Sobol sampling (https://pypi.org/project/sobol_seq/). 

You can install the above dependencies and pyrobust using pip, e.g.:

```
pip install pyrobust
```

## Example code

The code for the examples in this documentation can be found at https://github.com/EPCCed/pyrobust-examples.


## Bugs

Bugs can be reported [via Github](https://github.com/EPCCed/pynmmso/issues). We'd also be keen for feedback on this documentation and the examples provided.

## Credits

The following people have contributed to this project:

* Professor Jonathan Fieldsend, Computer Science, University of Exeter
* Dr Ozgur Akman, Mathematics, University of Exeter
* Dr Khulood Alyahya, Computer Science, University of Exeter
* Ally Hume, EPCC, University of Edinburgh
* Dr Chris Wood, EPCC, University of Edinburgh
* Dr Neelofer Banglawala, EPCC, University of Edinburgh
* Professor Andrew J Millar, Chair of Systems Biology, The University of Edinburgh
* Dr Kevin Doherty
* Benjamin J. Wareham

Thanks to the following tools used to produce the graphs on this documentation:

* pyDOE (https://pythonhosted.org/pyDOE/)
* sobol_seq (https://pypi.org/project/sobol_seq/)

This work was supported by the Engineering and Physical Sciences Research Council (grant number [EP/N018125/1](https://gow.epsrc.ukri.org/NGBOViewGrant.aspx?GrantRef=EP/N018125/1))
