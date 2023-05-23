#ifndef _FUNCTIONAL_CLASS_hpp_
#define _FUNCTIONAL_CLASS_hpp_

#include "1RDM_class.hpp"
#include <eigen3/unsupported/Eigen/CXX11/Tensor> 

using namespace std;
using namespace Eigen;

class RDM1;


VectorXd None (RDM1*, int);
//Definition of the Functional class (for more details see Functional_class.cpp)
class Functional{
    private:
        bool is_J_func_;                       //Wether the direct term is included in the functional
        VectorXd (*n_K_)(RDM1*);               //Function of the occupation in the K part of the 
                                               //energy (only needed for the analytical Hessian)
        VectorXd (*dn_K_)(RDM1*,int);          //Derivative of n_K_ (respect to x_i)
        double (*dn_Kn_)(RDM1*,int);           // ith element of dn_K_/dn
        VectorXd (*ddn_K_)(RDM1*,int,int);     //2nd derivative of n_K_
        double (*ddn_Kn_)(RDM1*,int);          // ith element od d^2n_K/dn^2  
    public:
        Functional(bool is_J_func = false, VectorXd (*)(RDM1*) = nullptr, VectorXd (*)(RDM1*,int) = nullptr,  double (*)(RDM1*,int) = nullptr, VectorXd (*)(RDM1*,int,int) = nullptr, double (*)(RDM1*,int) = nullptr);
        Functional(VectorXd (*)(RDM1*), VectorXd (*)(RDM1*,int), double (*)(RDM1*,int), VectorXd (*)(RDM1*,int,int), double (*)(RDM1*,int), bool is_J_func_ = false);
        bool needs_subspace() const;
        double E(RDM1*) const; VectorXd grad_E(RDM1*,bool only_n=false,bool only_no=false) const; MatrixXd hess_E(RDM1*,MatrixXd*,bool only_n=false,bool only_no=false) const; 
        VectorXd grad_aux(RDM1*,bool only_n= false, bool only_no= false); MatrixXd hess_aux(RDM1*, bool only_n=false, bool only_no=false) const;
        double E(RDM1*, MatrixXd*, MatrixXd*) const; VectorXd grad_E(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        double E_Hxc(MatrixXd*, MatrixXd*) const;
        VectorXd dE_Hxc(RDM1*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc(RDM1*, MatrixXd*, MatrixXd*, bool only_n=false, bool only_no=false) const;
        VectorXd dE_Hxc_aux(RDM1*, bool only_n= false, bool only_no= false);
        MatrixXd Jac(RDM1*,bool only_n=false, bool only_no=false) const;
        MatrixXd ddE_Hxc(RDM1*, bool only_n=false, bool only_no=false) const;
        MatrixXd ddE_known(RDM1*,bool only_n=false, bool only_no=false) const; 
        MatrixXd ddE_approx(RDM1*, MatrixXd*, bool only_n=false, bool only_no=false) const; 
        MatrixXd ddE_Hxc_aux(RDM1* gamma, bool only_n=false, bool only_no=false) const;       

        MatrixXd v_J(RDM1*) const ; MatrixXd v_K(RDM1*) const; 
        MatrixXd v_J_AO(RDM1*) const ; MatrixXd v_K_AO(RDM1*) const; 
        MatrixXd dv_J_n_old(RDM1*,int) const; MatrixXd dv_K_n_old(RDM1*,int) const; 
        MatrixXd dv_J_NO(RDM1*,int) const; MatrixXd dv_K_NO(RDM1*,int) const;
        
        MatrixXd compute_WJ(RDM1*) const;   MatrixXd compute_WK(RDM1*) const; 
        VectorXd compute_dW_J(RDM1*) const; VectorXd compute_dW_K(RDM1*) const; 
        VectorXd compute_dNO_J(RDM1*) const; VectorXd compute_dNO_K(RDM1*) const;

        MatrixXd dv_J_n (RDM1*,int) const; MatrixXd dv_K_n (RDM1*,int) const;
        Tensor<double,4> ddW_J_n(RDM1*) const; Tensor<double,4> ddW_K_n(RDM1*) const;
        double ddW_J_n(RDM1*,int,int) const; double ddW_K_n(RDM1*,int,int) const;
        MatrixXd ddv_J_NO(RDM1*,int,int) const; MatrixXd ddv_K_NO(RDM1*,int,int) const;
};
//Auxiliary functions to compute the energy
double E1(RDM1*);VectorXd dE1(RDM1*, bool only_n= false, bool only_no= false); VectorXd dE1_aux(RDM1*, bool only_n= false, bool only_no= false); 
MatrixXd ddE1(RDM1*, bool only_n= false, bool only_no= false); MatrixXd ddE1_aux(RDM1*, bool only_n, bool only_no);
double compute_E1(MatrixXd*, MatrixXd*); 
#endif