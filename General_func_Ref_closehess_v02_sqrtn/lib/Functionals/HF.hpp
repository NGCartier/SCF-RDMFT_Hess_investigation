#ifndef _HF_FUNC_hpp_
#define _HF_FUNC_hpp_

class Functional; class RDM1;
VectorXd HF_n_K(RDM1*); VectorXd HF_dn_K(RDM1*,int); double HF_dn_Kn(RDM1*,int); VectorXd HF_ddn_K(RDM1*,int,int); double HF_ddn_Kn(RDM1*,int);
static Functional HF_func = Functional(HF_n_K,HF_dn_K,HF_dn_Kn,HF_ddn_K,HF_ddn_Kn);
VectorXd H_n_K(RDM1*); VectorXd H_dn_K(RDM1*,int); double H_dn_Kn(RDM1*,int); VectorXd H_ddn_K(RDM1*,int,int); double H_ddn_Kn(RDM1*,int);
static Functional Hartree_func = Functional(H_n_K,H_dn_K,H_dn_Kn,H_ddn_K,H_ddn_Kn);
VectorXd E1_n_K(RDM1*); VectorXd E1_dn_K(RDM1*,int); double E1_dn_Kn (RDM1*,int); VectorXd E1_ddn_K(RDM1*,int,int); double E1_ddn_Kn(RDM1*,int);
static Functional E1_func = Functional(E1_n_K,E1_dn_K,E1_dn_Kn,E1_ddn_K,E1_ddn_Kn,true);
#endif