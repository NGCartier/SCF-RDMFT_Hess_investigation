#ifndef _FUNCTIONAL_CLASS_hpp_
#define _FUNCTIONAL_CLASS_hpp_

#include "1RDM_class.hpp"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
using namespace std;
using namespace Eigen;

class RDM1; 

//Definition of the Functional class (for more details see Functional_class.cpp)
class Functional{
    private:
        bool is_J_func_;                       //Wether the direct term is included in the functional
        MatrixXd (*W_K_) (RDM1*);              //The matrix of E_xc (or E_Hxc if is_J_func_=true) in NO basis.
        VectorXd (*dW_K_)(RDM1*);              //The derivative of W_K_
        VectorXd (*dW_K_subspace_)(RDM1*,int); //dW_K_ restricted to one subspace of PNOF omega (for occ only).
        VectorXd (*n_K_)(RDM1*);               //Function of the occupation in the K part of the 
                                               //energy (only needed for the analytical Hessian)
        VectorXd (*dn_K_)(RDM1*,int);          //Derivative of n_K_
        double   (*dn_Kn_)(RDM1*,int);         //ith element of dn_K_/dn
        VectorXd (*ddn_K_)(RDM1*,int,int);     //2nd derivative of n_K_
        double   (*ddn_Kn_)(RDM1*,int,int);     //ith and jth entry of d^2 n_K_/dn_i dn_j
    public:
        Functional(MatrixXd(*)(RDM1*), VectorXd(*)(RDM1*), bool is_J_func_ = false, VectorXd (*dW_K_subspace_)(RDM1*,int) = nullptr, VectorXd (*n_K_)(RDM1*) = nullptr, 
                    VectorXd (*dn_K_)(RDM1*,int) = nullptr, double (*dn_Kn_)(RDM1*,int) = nullptr, VectorXd (*ddn_K_)(RDM1*,int,int) = nullptr, double (*ddn_Kn_)(RDM1*,int,int) = nullptr);
        Functional(MatrixXd(*)(RDM1*), VectorXd(*)(RDM1*), VectorXd (*n_K_)(RDM1*), VectorXd (*dn_K_)(RDM1*,int), double (*dn_Kn_)(RDM1*,int), 
                    VectorXd (*ddn_K_)(RDM1*,int,int), double (*ddn_Kn_)(RDM1*,int,int), bool is_J_func_ = false);
        bool needs_subspace() const;
        double E(RDM1*) const; VectorXd grad_E(RDM1*,bool only_n=false,bool only_no=false) const; 
        double E(RDM1*, MatrixXd*, MatrixXd*) const; VectorXd grad_E(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        VectorXd grad_E_subspace(RDM1*, int) const;
        MatrixXd hess_E_exa(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const;
        MatrixXd hess_E(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const;
        MatrixXd hess_E_cheap(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const;    
        double E_Hxc(MatrixXd*, MatrixXd*) const;
        VectorXd dE_Hxc(RDM1*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc_subspace(RDM1*, int) const;
        MatrixXd ddE_Hxc(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const; MatrixXd ddE_Hxc_k(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const;
        MatrixXd ddE_Hxc_aux(RDM1*, bool only_n=false, bool only_no=false, bool only_coupled=false) const; 

        MatrixXd Jac(RDM1*,bool only_n=false,bool only_no=false) const;
        MatrixXd ddJac(RDM1*, bool only_n=false, bool only_no=false) const;
        MatrixXd dv_J(RDM1*,int) const; MatrixXd dv_K(RDM1*,int) const; 
        MatrixXd compute_WJ(RDM1*) const;    MatrixXd compute_WK(RDM1*) const; 
        Tensor<double,4> compute_Wbar_J(RDM1*) const; Tensor<double,4> compute_Wbar_K(RDM1*) const;
        VectorXd compute_dW_J(RDM1*) const;  VectorXd compute_dW_K(RDM1*) const;
        MatrixXd compute_ddW_J(RDM1*) const; MatrixXd compute_ddW_K(RDM1*) const; 
        MatrixXd x_space_hess(RDM1*,MatrixXd*) const;
        
};
//Auxiliary functions to compute the energy
double E1(RDM1*); double compute_E1(MatrixXd*, MatrixXd*); VectorXd dE1(RDM1*, bool only_n= false, bool only_no= false); 
MatrixXd ddE1(RDM1*, bool only_n= false, bool only_no= false); 
MatrixXd ddE1_k(RDM1*, bool only_n= false, bool only_no= false); 
MatrixXd v_J(RDM1*,VectorXd*, bool is_J_func = false);  MatrixXd v_K(RDM1*,VectorXd*); 

Tensor<double,4> Tensor_prod(MatrixXd, Tensor<double,4>, int, int);
MatrixXd dU(MatrixXd*,int,int); MatrixXd ddU(MatrixXd*,int,int,int,int);
VectorXd pow(const VectorXd*, double);

#endif