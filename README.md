# SCF-RDMFT_Hess_investigation
Modification of the SCF-RDMFT repository with exact derivative and different approximations of the Hessian. Used to test impact of the different Hessians.

After having compilde the Test.py file, to call a 1RDM optimisation I use the function\
compute_1RDM(mol, guess="CISD", func="Muller", disp=0, epsi = 1e-6, Maxiter=10000, 
                 hess_approx, file, cond='xtol' )\
Arguments:\
*gto from PySCF mol, the molecule onwhich the energy is computed (H2O or H2O_pvdz usually), a list of molecules is defined in Test.py,\
*string guess, the initial guess of the 1RDM, can be "CISD", "CCSD" or Hartree-Fock with Fermi-Dirac for the occupations (use "HF"),\
*string func, the functional to use, "Muller" of Hartree-Fock ("HF") with this code,\
*int disp, the level of detail of the output (=0 returns only the final energy, =1 also returns the number of iterations and time to do the *conputation, =3 returns some informations on the macro iterations),\
*double epsi, relative precision required on the energy for the algorithm to converge, \
*int Maxiter, the maximum number of iteration per macro iteration (I use a large one to ensure that it only triggers in extrem cases),\
*string hess_approx, describes the Hessian to use to do the optimisation (detail in classes/1RDM_class.cpp in the opti_aux function):
  - add BFGS or SR1 or exa to use the corresponding Hessian,
  - add _aux to the Hessian in the gamma space,
  - add _exastart to start with the exact Hessian,
  - add _nn, _NONO, _nNO to use the occ-occ, NO-NO, occ-NO block of the exact hessian (repsectively),
  - add hybrid_m, to use the exact Hessian every m iterations.
  ex : "BFGS_aux_nn_NONO" will use the exact Hessian at the for the occ-occ and NO-NO blocks and the BFGS Hessian for the occ-NO block in gamma space.\
*string file, path and name of the file where the output data will be writen :\
  this will creat 2 files, a file with the name provided and a file with an extra "_behaviour",\
  in the 1st file will be writen the energy and Hessian difference (with the exact one, total and per block) at each step,
  in the 2nd file the energy number of negative eigenvalues of the Hessian, difference of the exact Hessian between two steps, the step and gradient norms at each iteration.\
*string cond, the termination condition of a macro-iteration, 'xtol' for comvergence in the parameters 'gtol' for convergence in the gradient.

Return:\
a vector of the natural occupations and matrix of the natural orbitals. 

----------------------------------------------------------------------------

To plot the results you can use functions avalable in plot_fig_closfull_hess.py.
The path and name of the files will need to be addapted to your calls to compute_1RDM
