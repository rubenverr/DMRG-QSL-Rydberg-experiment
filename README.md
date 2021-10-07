# DMRG-QSL-Rydberg-experiment
Static and dynamic DMRG results reported in arxiv:2104.04119 and the codes used to obtain them.

###################################

Tensor network simulations were used for the following plots in the supplemental material of the aforementioned paper: Figures S13, S14, S16, S17.
Correspondingly, four files in this repository are spreadsheets containing the numerical values appearing in these plots.

There are also three python codes in this repository:
1) "rydberg_on_ruby.py": creates the model file for Rydberg atoms placed on links of kagome lattice (i.e., ruby lattice with rho=sqrt(3))
2) "get_gs.py": obtains the ground state for the aforementioned model using DMRG, and various physical quantities are calculated
3) "dynamic_state_prep_on_ruby_lattice.py": time-evolves with a time-dependent Rydberg Hamiltonian, mimicking experimental state-preparation

All these python files use the tenpy library: see https://github.com/tenpy/tenpy or https://tenpy.readthedocs.io/
The simulations for arxiv:2104.04119 were performed on version "tenpy 0.7.2".
