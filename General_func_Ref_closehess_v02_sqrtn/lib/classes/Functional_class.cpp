#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <fstream>
#include <iostream>
#include <time.h>

using namespace std;
using namespace Eigen;

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "Matrix_Tensor_converter.cpp"
#include "EBI_add.hpp"

/*Constructor for the Functional class
\param args 8 functions defining the functional F, assumed to be of the form:
        F = WJ(fJ,gJ) -WK(fK,gK) where WJ(fJ,gJ) = sum_{mu nu kappa lambda}fJ(n)_{mu nu}gJ(n)_{lambda kappa}[mu nu|kappa lambda]
                                       WK(fK,gK) = sum-{mu nu kappa lambda}fK(n)_{lambda mu}gK(n)_{nu kappa}[mu nu|kappa lambda]
                                       (n the occupations, J Coulomb, K exchange)
        see K.J.H. "Giesbertz, Avoiding the 4-index transformation in one-body reduced density matrix functional calculations for 
            separable functionals", Phys. Chem. Chem. Phys. 18, 21024-21031 (2016)
*/

Functional::Functional(bool is_J_func, VectorXd (*n_K)(RDM1*), VectorXd (*dn_K)(RDM1*,int), double (*dn_Kn)(RDM1*,int), VectorXd (*ddn_K)(RDM1*,int,int), double (*ddn_Kn)(RDM1*,int)){
    is_J_func_ = is_J_func; n_K_ = n_K; dn_K_ = dn_K; dn_Kn_ = dn_Kn; ddn_K_ = ddn_K; ddn_Kn_ = ddn_Kn;
}
Functional::Functional(VectorXd (*n_K)(RDM1*), VectorXd (*dn_K)(RDM1*,int), double (*dn_Kn)(RDM1*,int), VectorXd (*ddn_K)(RDM1*,int,int), double (*ddn_Kn)(RDM1*,int), bool is_J_func){
    n_K_ = n_K; dn_K_ = dn_K; dn_Kn_ = dn_Kn; ddn_K_ = ddn_K; ddn_Kn_ = ddn_Kn; is_J_func_ = is_J_func;
}

//Check wether functional needs to build the subspaces (i.e. if functionals is PNOF7 for now)
bool Functional::needs_subspace() const{
    return false; //to adapt if want to use PNOF-like functionals 
}

//Computes the energy of gamma for the functional
double Functional::E(RDM1* gamma) const {
    int l = gamma->size();
    MatrixXd W_J(l,l); W_J = compute_WJ(gamma); MatrixXd W_K(l,l); W_K = compute_WK(gamma);
    return E1(gamma) + E_Hxc(&W_J,&W_K);
}

double Functional::E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K) const {
    return E1(gamma) + E_Hxc(W_J,W_K);
}

//Computes the gradient of the energy of gamma for the functional
//if only_n = False return the derivatives respect to the NOs
//if only_no = False return the derivatives respect to the occupations
VectorXd Functional::grad_E(RDM1* gamma, bool only_n, bool only_no) const {
    MatrixXd W_J = compute_WJ(gamma); MatrixXd W_K = compute_WK(gamma);
    dE1(gamma, only_n, only_no);
    dE_Hxc(gamma, &W_J,&W_K, only_n, only_no);
    return dE1(gamma, only_n, only_no)+dE_Hxc(gamma, &W_J,&W_K, only_n, only_no);
}

VectorXd Functional::grad_E(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const {
    return dE1(gamma, only_n, only_no)+dE_Hxc(gamma, W_J, W_K, only_n, only_no);
}

//Compute Hessian of the energy of gamma for the functional using the approximate Hessian of E (respect to U/n)
MatrixXd Functional::hess_E(RDM1* gamma, MatrixXd* H_approx, bool only_n, bool only_no) const{
    return ddE_known(gamma, only_n, only_no) + ddE_approx(gamma,H_approx,only_n,only_no); 
}

//Computes the functional independant part of the energy (1 electron and nuclei)
double E1(RDM1* gamma){
    MatrixXd g = gamma->mat();
    return gamma->E_nuc + compute_E1(&gamma->int1e,&g);
}

//Computes the derivative of the 1 electron part of the energy
VectorXd dE1(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int ll = l*(l+1)/2;
    MatrixXd* H1 = &gamma->int1e ; MatrixXd N = gamma->n().asDiagonal(); 
    VectorXd dE1 (ll); MatrixXd g = gamma->mat();
    if (not only_no){
        for (int i =0; i<l;i++){
            MatrixXd dg = gamma->dmat(i);
            dE1(i) = compute_E1(H1,&dg);
        } 
    }
    if (not only_n){
        int index = l;
        for (int i =l; i<ll;i++){
            MatrixXd dg = gamma->dmat(i);
            dE1(i) =  compute_E1(H1,&dg);
        }
    }
    if (only_n) {return dE1.segment(0,l);}
    else{ if(only_no){return dE1.segment(l, l*(l-1)/2);}
    else{ return dE1;}
    }
    
}

// Compute the 1 electron part of the energy
double compute_E1(MatrixXd* H, MatrixXd* g){
    
    int l = g->rows(); int ll = pow(l,2);
    MatrixXd res =  H->reshaped(1,ll) * g->reshaped(ll,1);
    return res(0,0);

}
// Compute the potential in NO basis for J and K 
MatrixXd Functional::v_J(RDM1* gamma) const {
    int l = gamma->size(); int ll = pow(l,2);
    if(is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    else{
        MatrixXd no = gamma->no(); MatrixXd n = gamma->n().asDiagonal();
        MatrixXd f = no * n *no.transpose();
        MatrixXd res = ( gamma->int2e *f.reshaped(ll,1) ).reshaped(l,l);
        return no.transpose() *res *no;
    }
    
}

MatrixXd Functional::v_K(RDM1* gamma) const { 
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd no = gamma->no(); MatrixXd n = n_K_(gamma).asDiagonal();
    MatrixXd f = no * n * no.transpose();
    MatrixXd res = (gamma->int2e_x * f.reshaped(ll,1)).reshaped(l,l);
    return no.transpose() * res * no;
}

MatrixXd Functional::v_J_AO(RDM1* gamma) const {
    int l = gamma->size(); int ll = pow(l,2);
    if(is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    else{
        MatrixXd no = gamma->no(); MatrixXd n = gamma->n().asDiagonal();
        MatrixXd f = no* n *no.transpose();
        MatrixXd res = ( gamma->int2e *f.reshaped(ll,1) ).reshaped(l,l);
        return res;
    }
    
}
MatrixXd Functional::v_K_AO(RDM1* gamma) const { 
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd no = gamma->no(); MatrixXd n = n_K_(gamma).asDiagonal();
    MatrixXd f = no * n * no.transpose();
    MatrixXd res = (gamma->int2e_x * f.reshaped(ll,1)).reshaped(l,l);
    return res;
}

// Compute the derivative of v_J, v_K respect to the NOs 
MatrixXd Functional::dv_J_NO(RDM1* gamma, int i) const {
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd no = gamma->no(); MatrixXd dno = gamma->dno(i); MatrixXd n = gamma->n().asDiagonal();
    MatrixXd f = dno * n *no.transpose() + no * n *dno.transpose();
    MatrixXd res = ( gamma->int2e * f.reshaped(ll,1) ).reshaped(l,l);
    return no.transpose() *res *no;
}
MatrixXd Functional::dv_K_NO(RDM1* gamma, int i) const {
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd no = gamma->no(); MatrixXd dno = gamma->dno(i); MatrixXd n = n_K_(gamma).asDiagonal();
    MatrixXd df = dno*n*no.transpose() + no*n*dno.transpose();
    MatrixXd res = ( gamma->int2e_x * df.reshaped(ll,1) ).reshaped(l,l);
    return no.transpose() *res *no;
}

// Compute the derivative of v_J, v_K respect to the params of the occupations
MatrixXd Functional::dv_J_n_old(RDM1* gamma,int i) const{
    int l = gamma->size(); int ll = pow(l,2);
    if (is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    MatrixXd no = gamma->no();
    MatrixXd df = no * gamma->dn(i).asDiagonal() * no.transpose();
    MatrixXd res = (gamma->int2e * df.reshaped(ll,1)).reshaped(l,l);
    return no.transpose() * res * no;
}

MatrixXd Functional::dv_K_n_old(RDM1* gamma, int i) const{ 
    int l = gamma->size(); int ll = pow(l,2);
    MatrixXd no = gamma->no();
    MatrixXd df = no * dn_K_(gamma,i).asDiagonal() * no.transpose();
    MatrixXd res = (gamma->int2e_x * df.reshaped(ll,1)).reshaped(l,l);
    return no.transpose() * res * no;
}

// Compute the 1st and 2nd derivative of v_J, v_K respect to the occupations
// Compute the 2 electron integral in NO basis 
MatrixXd Functional::dv_J_n(RDM1* gamma, int i) const {
    int l = gamma->size(); int ll = l*l;
    if (is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    MatrixXd no = gamma->no(); 
    MatrixXd delta = MatrixXd::Zero(l,l); delta(i,i) = 2.*gamma->sqrtn(i);
    MatrixXd f = no * delta *no.transpose();
    MatrixXd res = ( gamma->int2e * f.reshaped(ll,1) ).reshaped(l,l);
    return no.transpose() *res *no;
}

MatrixXd Functional::dv_K_n(RDM1* gamma, int i) const {
    int l = gamma->size(); int ll = l*l;
    MatrixXd no = gamma->no(); 
    MatrixXd delta = MatrixXd::Zero(l,l); delta(i,i) = dn_Kn_(gamma,i);
    MatrixXd f = no * delta *no.transpose();
    MatrixXd res = ( gamma->int2e_x * f.reshaped(ll,1) ).reshaped(l,l);
    return no.transpose() *res *no;
}

Tensor<double,4> Functional::ddW_J_n(RDM1* gamma) const {
    //Doesn't work. contract does not return expected result
    int l = gamma->size(); Tensor<double,4> res(l,l,l,l);
    if (is_J_func_){
        res.setZero(); return res;
    }
    res = TensorCast(gamma->int2e,l,l,l,l);
    Tensor<double,2> no = TensorCast(gamma->no(),l,l);
    //The order of contractions matters -> to look at => does not seem to work correctly
    Eigen::array<IndexPair<int>,1> index = {IndexPair<int>(3,0)}; Tensor<double,4> temp = res.contract(no,index);
    index = {IndexPair<int>(2,1)}; res = temp.contract(no,index);
    index = {IndexPair<int>(1,0)}; temp = res.contract(no,index);
    index = {IndexPair<int>(0,1)}; res = temp.contract(no,index);
    return res;
}

double Functional::ddW_J_n(RDM1* gamma, int i, int j) const {
    int l = gamma->size(); int ll = l*l;
    if (is_J_func_){
        return 0;
    }
    MatrixXd no = gamma->no();
    MatrixXd delta_i = MatrixXd::Zero(l,l); delta_i(i,i) = 2.*gamma->sqrtn(i);
    MatrixXd delta_j = MatrixXd::Zero(l,l); delta_j(j,j) = 2.*gamma->sqrtn(j);
    MatrixXd df_i = no * delta_i * no.transpose();
    MatrixXd df_j = no * delta_j * no.transpose();
    MatrixXd res = (df_i.reshaped(1,ll) * gamma->int2e * df_j.reshaped(ll,1));
    return res(0,0);
}

Tensor<double,4> Functional::ddW_K_n(RDM1* gamma) const {
    //Doesn't work. contract does not return expected result
    int l = gamma->size();
    Tensor<double,4> res = TensorCast(gamma->int2e_x,l,l,l,l);
    Tensor<double,2> no = TensorCast(gamma->no(),l,l);
    //The order of contractions matters -> to look at => does not seem to work correctly
    Eigen::array<IndexPair<int>,1> index = {IndexPair<int>(3,0)}; Tensor<double,4> temp = res.contract(no,index);
    index = {IndexPair<int>(2,1)}; res = temp.contract(no,index);
    index = {IndexPair<int>(1,0)}; temp = res.contract(no,index);
    index = {IndexPair<int>(0,1)}; res = temp.contract(no,index);
    return 2.*res;
}

double Functional::ddW_K_n(RDM1* gamma, int i, int j) const {
    int l = gamma->size(); int ll = l*l;
    MatrixXd no = gamma->no();
    MatrixXd delta_i = MatrixXd::Zero(l,l); delta_i(i,i) = dn_Kn_(gamma,i);
    MatrixXd delta_j = MatrixXd::Zero(l,l); delta_j(j,j) = dn_Kn_(gamma,j);
    MatrixXd df_i = no * delta_i * no.transpose();
    MatrixXd df_j = no * delta_j * no.transpose();
    MatrixXd res = (df_i.reshaped(1,ll) * gamma->int2e_x * df_j.reshaped(ll,1));
    return res(0,0);
}
// Compute the 2nd derivative of v_J, v_K respect to the NO entries
// Rmk: tensor based version would be faster but see issue above

MatrixXd Functional::ddv_J_NO(RDM1* gamma, int i, int j) const {
    int l = gamma->size(); int ll = l*l;
    if (is_J_func_){
        return MatrixXd::Zero(l,l);
    }
    MatrixXd no = gamma->no(); MatrixXd no0 = gamma->no0(); MatrixXd n = gamma->n().asDiagonal(); 
    MatrixXd delta_ij = MatrixXd::Zero(l,l); delta_ij(i,j) =1.;
    MatrixXd df_ij = no0 * delta_ij * n * no.transpose();
    MatrixXd res = (df_ij.reshaped(1,ll)  * gamma->int2e).reshaped(l,l) ;
    return 2.* no.transpose() * res * no;
}

MatrixXd Functional::ddv_K_NO(RDM1* gamma, int i, int j) const {
    int l = gamma->size(); int ll = l*l;
    MatrixXd no = gamma->no(); MatrixXd no0 = gamma->no0(); MatrixXd n = n_K_(gamma).asDiagonal(); 
    MatrixXd delta_ij = MatrixXd::Zero(l,l); delta_ij(i,j) =1.;
    MatrixXd df_ij = no0 * delta_ij * n * no.transpose();
    MatrixXd res = (df_ij.reshaped(1,ll) * gamma->int2e_x ).reshaped(l,l);
    return 2.* no.transpose() * res * no;
}

// Compute the W_J W_K matrices in NO basis
MatrixXd Functional::compute_WJ(RDM1* gamma) const{
    int l = gamma->size(); 
    if (is_J_func_){ 
        return MatrixXd::Zero(l,l);
    }
    MatrixXd W (l,l);
    VectorXd n = gamma->n(); MatrixXd v = v_J(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = n(i)* v(i,j);
        }
    }
    return W;
}

MatrixXd Functional::compute_WK(RDM1* gamma) const{
    int l = gamma->size(); 
    MatrixXd W (l,l);
    VectorXd n = n_K_(gamma); MatrixXd v = v_K(gamma);
    for (int i = 0; i<l; i++){
        for (int j = 0; j<l; j++){
            W(i,j) = n(i)* v(i,j);
        }
    }
    return W;
}
// Compute the derivative of W_J W_K respect to the occupations
VectorXd Functional::compute_dW_J(RDM1* gamma) const{
    int l = gamma->size();
    if (is_J_func_){
        return VectorXd::Zero(l);
    }
    else{
        VectorXd dW_J = VectorXd::Zero(l);
        VectorXd v = v_J(gamma).diagonal();
        for (int i=0;i<l;i++){
            VectorXd dn = gamma->dn(i);
            dW_J(i) = dn.dot(v);
        } 
        return dW_J;
    }
}

VectorXd Functional::compute_dW_K(RDM1* gamma) const{
    int l = gamma->size();
    VectorXd dW_K = VectorXd::Zero(l);
    VectorXd v = v_K(gamma).diagonal();
    for (int i=0;i<l;i++){
        VectorXd dn = dn_K_(gamma,i);
        dW_K(i) = dn.dot(v);
    } 
    return dW_K;
}

// Compute the derivative of W_J W_K respect to the NOs
VectorXd Functional::compute_dNO_J(RDM1* gamma) const{
    int l = gamma->size(); int ll = l*(l-1)/2;
    if (is_J_func_){
        return VectorXd::Zero(ll);
    }
    else{
        VectorXd dW_J (ll);
        for (int i=0;i<ll;i++){
            VectorXd N = gamma->n();
            VectorXd dv = dv_J_NO(gamma,i).diagonal();
            dW_J(i)= N.dot(dv);
        }
        return dW_J;
    }
}

VectorXd Functional::compute_dNO_K(RDM1* gamma) const{
    int l = gamma->size(); int ll = l*(l-1)/2;
    VectorXd dW_K (ll);
    for (int i=0;i<ll;i++){
        VectorXd N = n_K_(gamma);
        VectorXd dv = dv_K_NO(gamma,i).diagonal();
        dW_K(i)= N.dot(dv);
    }
    return dW_K;
}

// Compute the Hatree exchange correlation energy  
double Functional::E_Hxc(MatrixXd* W_J, MatrixXd* W_K) const{
    return 1./2.*( (*W_J).trace()-(*W_K).trace() );
}

// Compute the gradiant of the Hatree exchange correlation energy
VectorXd Functional::dE_Hxc(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        dE2.segment(l,ll-l) = compute_dNO_J(gamma) - compute_dNO_K(gamma);
    }
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){return dE2.segment(l,ll-l);}
    else{return dE2;}}
}

VectorXd Functional::dE_Hxc(RDM1* gamma, MatrixXd* W_J, MatrixXd* W_K, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd dE2 (ll);  
    if (not only_no){
        dE2.segment(0,l) = compute_dW_J(gamma) - compute_dW_K(gamma);
    }
    if (not only_n){
        dE2.segment(l,ll-l) = compute_dNO_J(gamma) - compute_dNO_K(gamma);
    }
    
    if(only_n){ return dE2.segment(0,l);}
    else{if(only_no){ return dE2.segment(l,ll-l);}
    else{return dE2;}}
}

// Computation of the gradient used for the approximation of the auxiliary Hessian
VectorXd Functional::grad_aux(RDM1* gamma, bool only_n, bool only_no){
    return dE1_aux(gamma, only_n, only_no) + dE_Hxc_aux(gamma, only_n, only_no);
}

VectorXd dE1_aux(RDM1* gamma,bool only_n, bool only_no){
    int l = gamma->size(); int l2 = l*l; VectorXd dE1 (l2+l);
    MatrixXd NO = gamma->no();
    MatrixXd H1 = NO.transpose()*gamma->int1e*NO;
    if(not only_no){
        
        dE1.segment(0,l) = 2.*gamma->sqrtn().cwiseProduct( H1.diagonal() );
    }
    if(not only_n){
        MatrixXd U = Givens(MatrixXd::Identity(l,l),&gamma->l_theta);
        dE1.segment(l,l2) = 2.*(U*H1*gamma->n().asDiagonal()).transpose().reshaped(1,l2).row(0);
    }
    if(only_n){return dE1.segment(0,l);}
    else{ if(only_no){return dE1.segment(l,l2);}
    else{ return dE1;}
    }
}

VectorXd Functional::dE_Hxc_aux(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int l2 = l*l; VectorXd dE2 (l2+l);
    MatrixXd vJ = v_J(gamma);
    MatrixXd vK = v_K(gamma);
    if (not only_no){
        
        VectorXd dn_K (l);
        for(int i=0;i<l;i++){
            dn_K(i) = dn_Kn_(gamma,i);
        }
        dE2.segment(0,l) = 2.*gamma->sqrtn().cwiseProduct(vJ.diagonal()) 
                            - dn_K.cwiseProduct(vK.diagonal());
    }
    if (not only_n){
        MatrixXd U = Givens(MatrixXd::Identity(l,l),&gamma->l_theta);
        MatrixXd n_J = gamma->n().asDiagonal();
        MatrixXd dE2_J = 2.*U*vJ*n_J;
        MatrixXd n_K = n_K_(gamma).asDiagonal();
        MatrixXd dE2_K = 2.*U*vK*n_K; 
        dE2.segment(l,l2) = (dE2_J-dE2_K).transpose().reshaped(1,l2).row(0);
    }
    if(only_n){return dE2.segment(0,l);}
    else{ if(only_no){return dE2.segment(l,l2);}
    else{ return dE2;}
    }
}

// Compute the dU/dx d2E/dU2 dU/dx term of the energy Hessian
MatrixXd Functional::Jac(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 =l*l; MatrixXd I = MatrixXd::Identity(l,l);
    MatrixXd J = MatrixXd::Zero(l2+l,ll);
    if (not only_n){
        for(int i=0;i<l;i++){
            J.block(0,i,l,1) = gamma->dsqrt_n(i);
        }
    }
    if (not only_no){
        for(int i=l;i<ll;i++){
            J.block(l,i,l2,1) = dGivens(I,&gamma->l_theta,i-l).transpose().reshaped(l2,1);
        }
    }
    if(only_n){return J.block(0,0,l,l);}
    else if(only_no){return J.block(l,l,l2,ll-l);}
    else {return J;}
}

MatrixXd Functional::ddE_approx(RDM1* gamma, MatrixXd* H_approx, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 =l*l; MatrixXd I = MatrixXd::Identity(l,l);
    MatrixXd J = Jac(gamma,only_n,only_no); MatrixXd Jt = J.transpose();
    MatrixXd ddE = Jt*(*H_approx)*J;
    if (only_n){return ddE.block(0,0,l,l);}
    else{ if(only_no){return ddE.block(l,l,ll-l,ll-l);}
    else{ return ddE; }  
    }
}

// Compute the d2(U/n)/dx2 dE/(U/n) term of the energy Hessian
MatrixXd Functional::ddE_known(RDM1* gamma,bool only_n, bool only_no) const{
    return ddE1(gamma,only_n,only_no) + ddE_Hxc(gamma,only_n,only_no);
}

MatrixXd ddE1(RDM1* gamma,bool only_n, bool only_no){
    int l = gamma->size(); int ll = l*(l+1)/2; 
    MatrixXd NO = gamma->no();
    MatrixXd H1 = NO.transpose()*gamma->int1e*NO ; MatrixXd N = gamma->n().asDiagonal(); 
    MatrixXd ddE1 = MatrixXd::Zero(ll,ll); 
    if (not only_no){
        VectorXd dnE1 (l); 
        for(int i=0;i<l;i++){
            dnE1(i) = 2.*gamma->sqrtn(i)*H1(i,i);
        }
        for (int i=0;i<l;i++){
            for (int j=0;j<=i;j++){
                ddE1(i,j) = gamma->ddsqrt_n(i,j).dot(dnE1);
                ddE1(j,i) = ddE1(i,j);
            }
        }
    }
    if (not only_n){
        MatrixXd I = MatrixXd::Identity(l,l);
        MatrixXd U = Givens(I,&gamma->l_theta);
        MatrixXd dE1 = 2.*U*H1*gamma->n().asDiagonal(); dE1 = dE1.reshaped(l*l,1);
        for (int i=l;i<ll;i++){
            for (int j=l;j<=i;j++){
                MatrixXd ddU = ddGivens(I,&gamma->l_theta,i-l,j-l).reshaped(1,l*l);
                ddE1(i,j) = (ddU*dE1)(0,0); 
                ddE1(j,i) = ddE1(i,j);
            }
        } 
    }
    if (only_n){return ddE1.block(0,0,l,l);}
    else{ if(only_no){return ddE1.block(l,l,ll-l,ll-l);}
    else{ return ddE1; }
    }

}

MatrixXd Functional::ddE_Hxc(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd ddE2 = MatrixXd::Zero(ll,ll);
    MatrixXd vJ = v_J(gamma);
    MatrixXd vK = v_K(gamma);
    if (not only_no){
        VectorXd dnvJ (l); VectorXd dnvK (l);
        for(int i=0;i<l;i++){
            dnvJ(i) = 2.*gamma->sqrtn(i)*vJ(i,i);
            dnvK(i) = dn_Kn_(gamma,i)*vK(i,i);
        }
        for(int i=0;i<l;i++){
            for(int j=0;j<=i;j++){
                ddE2(i,j) = gamma->ddsqrt_n(i,j).dot(dnvJ - dnvK);
                ddE2(j,i) = ddE2(i,j);
            }
        }
    }
    if(not only_n){
        MatrixXd U = Givens(MatrixXd::Identity(l,l),&gamma->l_theta);
        MatrixXd dE2_J = 2.*U*vJ*gamma->n().asDiagonal();
        MatrixXd dE2_K = 2.*U*vK*n_K_(gamma).asDiagonal(); 
        MatrixXd dE2 = (dE2_J-dE2_K).reshaped(l*l,1);
        for(int i=l;i<ll;i++){
            for(int j=l;j<=i;j++){
                MatrixXd ddU = ddGivens(MatrixXd::Identity(l,l),&gamma->l_theta,i-l,j-l).reshaped(1,l*l);
                ddE2(i,j) = (ddU*dE2)(0,0);
                ddE2(j,i) = ddE2(i,j);
            }
        }
    }
    if(only_n){ return ddE2.block(0,0,l,l);}
    else{if(only_no){return ddE2.block(l,l,ll-l,ll-l);}
    else{ return ddE2;}}
}

// Computes the exact d2E/d(U,n) part (for exact Hessian)
MatrixXd Functional::hess_aux(RDM1* gamma, bool only_n, bool only_no) const{
    return ddE1_aux(gamma, only_n, only_no) + ddE_Hxc_aux(gamma, only_n, only_no);
}

MatrixXd ddE1_aux(RDM1* gamma, bool only_n, bool only_no){
    int l = gamma->size(); int l2 = l*l; MatrixXd ddE1 = MatrixXd::Zero(l2+l,l2+l) ;
    MatrixXd U = Givens(MatrixXd::Identity(l,l),&gamma->l_theta);
    MatrixXd H1 = gamma->no0().transpose()*gamma->int1e*gamma->no0();
    MatrixXd H1U = H1*U; MatrixXd UH1U = U.transpose()*H1U;
    if(not only_no){
        for (int i=0;i<l;i++){
            ddE1(i,i) = 2.*UH1U(i,i);
        }
    }
    if(not only_n){
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                for (int p=0;p<=i;p++){
                    int id1 = (i+1)*l+j; int id2 = (p+1)*l+j;
                    ddE1(id1,id2) = 2.*gamma->n(j)*H1(i,p);
                    ddE1(id2,id1) = ddE1(id1,id2);
                }
            }
        }
    }
    if( not only_n && not only_no){
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                int id = (i+1)*l+j;
                ddE1(id,j) = 4.*H1U(i,j)*gamma->sqrtn(j);
                ddE1(j,id) = ddE1(id,j);
            }
        }
    }
    if(only_n){return ddE1.block(0,0,l,l);}
    else{ if(only_no){return ddE1.block(l,l,l2,l2);}
    else{ return ddE1;}
    }
}

MatrixXd Functional::ddE_Hxc_aux(RDM1* gamma, bool only_n, bool only_no) const{
    int l = gamma->size(); int l2 = l*l; MatrixXd ddE2 = MatrixXd::Zero(l2+l,l2+l);
    MatrixXd vJ; MatrixXd vK; MatrixXd U; MatrixXd n; MatrixXd nK;
    if (not only_no){ 
        vJ = v_J(gamma);
        vK = v_K(gamma);
        for (int i=0;i<l;i++){
            for (int j=0;j<=i;j++){ 
                ddE2(i,j) = ddW_J_n(gamma,i,j) + 2.*(i==j)*vJ(i,i)
                            - ddW_K_n(gamma,i,j) - (i==j)*vK(i,i)*ddn_Kn_(gamma,i);
                ddE2(j,i) = ddE2(i,j);
            }
        }
    }
    if (not only_n){
        //could improve using more explicite formula, if we can use tensors : see notes
        vJ = gamma->no0().transpose()*v_J_AO(gamma)*gamma->no0(); 
        vK = gamma->no0().transpose()*v_K_AO(gamma)*gamma->no0(); 
        U = Givens(MatrixXd::Identity(l,l),&gamma->l_theta);
        n = gamma->n().asDiagonal(); nK = n_K_(gamma).asDiagonal();
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                int id1 = (i+1)*l+j; 
                MatrixXd ddvJ = ddv_J_NO(gamma,i,j); MatrixXd ddvK = ddv_K_NO(gamma,i,j);
                ddE2.block(id1,l,1,l2) += ( U*(2*ddvJ*n - (ddvK+ddvK.transpose())*nK) ).transpose().reshaped(1,l2);
                for (int p=0;p<=i;p++){
                    int id2 = (p+1)*l+j;
                    double temp = 2.*n(j,j)* vJ(i,p) - 2.* nK(j,j)* vK(i,p);
                    ddE2(id1,id2) += temp; if (id1!=id2){ddE2(id2,id1) += temp;}
                }
            }
        }
    }
    if( not only_n && not only_no){
        MatrixXd vJU = vJ*U;
        MatrixXd vKU = vK*U;
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                int id = (i+1)*l+j;
                ddE2(id,j) += 4.*vJU(i,j)*gamma->sqrtn(j) - 2.*vKU(i,j)*dn_Kn_(gamma,j);
                for (int p=0;p<l;p++){
                    ddE2(id,p) += (2.*U*(dv_J_n(gamma,p)*n - dv_K_n(gamma,p)*nK))(i,j); 
                    ddE2(p,id) = ddE2(id,p);
                }
            }
        }
    }
    if(only_n){return ddE2.block(0,0,l,l);}
    else{ if(only_no){return ddE2.block(l,l,l2,l2);}
    else{ return ddE2;}
    }
}