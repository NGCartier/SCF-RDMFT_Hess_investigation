#ifndef _MULLER_FUNC_hpp_
#define _MULLER_FUNC_hpp_
class Functional; class RDM1;
MatrixXd Muller_WK(RDM1*); VectorXd Muller_dWK(RDM1*); VectorXd Muller_n_K(RDM1*); VectorXd Muller_dn_K(RDM1*,int); 
double Muller_dn_Kn(RDM1*,int); VectorXd Muller_ddn_K(RDM1*,int,int); double Muller_ddn_Kn(RDM1*,int,int); 
static Functional Muller_func = Functional(Muller_WK, Muller_dWK, Muller_n_K, Muller_dn_K, Muller_dn_Kn, Muller_ddn_K,Muller_ddn_Kn);

#endif