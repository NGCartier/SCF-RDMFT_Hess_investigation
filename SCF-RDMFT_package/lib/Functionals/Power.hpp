#ifndef _POWER_FUNC_hpp_
#define _POWER_FUNC_hpp_
class Functional; class RDM1;
const double ALPHA = 0.7;
MatrixXd Power_WK(RDM1*); VectorXd Power_dWK(RDM1*); VectorXd Power_n_K(RDM1*); VectorXd Power_dn_K(RDM1*,int); 
double Power_dn_Kn(RDM1*,int); VectorXd Power_ddn_K(RDM1*,int,int); double Power_ddn_Kn(RDM1*,int,int); 
static Functional Power_func = Functional(Power_WK, Power_dWK, Power_n_K, Power_dn_K, Power_dn_Kn, Power_ddn_K,Power_ddn_Kn);

#endif