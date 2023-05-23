#ifndef _NUM_DERIVE_
#define _NUM_DERIVE_
class Functional; class RDM1;

Eigen::VectorXd grad_func(Functional* func, RDM1* gamma, bool do_n=true, bool do_no=true, double epsi=1e-6);
Eigen::MatrixXd hess_func(Functional* func, RDM1* gamma, bool do_n=true, bool do_no=true, double epsi=1e-5);
Eigen::VectorXd grad_func_aux(Functional* func, RDM1* gamma, bool do_n=true, bool do_no=true, double epsi=1e-6);
Eigen::MatrixXd hess_func_aux(Functional* func, RDM1* gamma, bool do_n=true, bool do_no=true, double epsi=8e-5);

#endif