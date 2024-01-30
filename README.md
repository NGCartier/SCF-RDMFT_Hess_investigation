# SCF-RDMFT_Hess_investigation
Modification of the SCF-RDMFT repository with exact and different approximations of the Hessian. Used to test impact of the different Hessian approximations.

To build the library use makefile,
Requires: 
  - C++ 17,
  - Python 3.8 and pybind11, for interface with Python,
  - PySCF for 1 and 2 electron integrals
  - fmt libaray for formatting, 
  - GSL libaray for math functions,
  - Eigen 3, for matrix library,
  - LAPACK and BLAS, to speed up Eigen.
Uses fides-cpp (https://github.com/dweindl/fides-cpp).
----

The function of interest is defined in Interface.py:
compute_1RDM(mol, guess="HF", func="Muller", disp=0, epsi = 1e-8, Maxiter=1000,hess_approx, file)\
Arguments:\
-gto from PySCF mol, the molecule on which the energy is computed, a list of molecules is defined in Test.py,\
-string guess, the initial guess of the 1RDM, can be "CISD" or Hartree-Fock with Fermi-Dirac for the occupations (use "HF"),\
-string func, the functional to use, "Muller" of Hartree-Fock ("HF") with this code,\
-int disp, the level of detail of the output (=0 returns only the final energy, =1 also returns the number of iterations and time to do the -conputation, =3 returns some informations on the macro iterations),\
-double epsi, relative precision required on the step and energy difference for the algorithm to converge, \
-int Maxiter, the maximum number of iteration per macro iteration,\
-string hess_approx, defines the expesnive part of the Hessian to use during the optimisation (detail in classes/1RDM_class.cpp in the opti_aux function):
  - use BFGS, DFP, SR1, exa (exact Hessian) or Zero (reduce the Hessian to the cheap part) to use the corresponding Hessian,
  - add _aux to the Hessian in the NU space,it is then possible to use tBFGS or sBFGS,
  - add _exastart to start with the exact Hessian,
  - add _nn, _NONO, _nNO to use the occ-occ, NO-NO, occ-NO block of the exact hessian (repsectively),
  - add hybrid_m, to use the exact Hessian every m iterations.
  
-string file, path and name of the file where the output data will be writen :\
  the energy, iteration, norm of the gradient and step will be reported in the file (other outputs are commented, see classes/1RDM.cpp) 
  
Return:\
a vector of the natural occupations and matrix of the natural orbitals. \
Print:\
Ground state energy obtained.

----
