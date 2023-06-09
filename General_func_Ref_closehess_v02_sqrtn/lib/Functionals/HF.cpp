#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "HF.hpp"

VectorXd HF_n_K(RDM1* gamma){
    return RSQRT_TWO*gamma->n();
}
VectorXd HF_dn_K(RDM1* gamma, int i){
    return RSQRT_TWO*gamma->dn(i);
}
double HF_dn_Kn(RDM1* gamma, int i){
    return 2.*RSQRT_TWO*gamma->sqrtn(i);
}
VectorXd HF_ddn_K(RDM1* gamma, int i, int j){
    return RSQRT_TWO*gamma->ddn(i,j);
}
double HF_ddn_Kn(RDM1* gamma, int i){
    return 2.*RSQRT_TWO;
}

VectorXd H_n_K(RDM1* gamma){
    return VectorXd::Zero(gamma->size());
}
VectorXd H_dn_K(RDM1* gamma,int i){
    return VectorXd::Zero(gamma->size());
}
double H_dn_Kn(RDM1* gamma,int i){
    return 0;
}
VectorXd H_ddn_K(RDM1* gamma,int i, int j){
    return VectorXd::Zero(gamma->size());
}
double H_ddn_Kn(RDM1* gamma, int i){
    return 0;
}

VectorXd E1_n_K(RDM1* gamma){
    return VectorXd::Zero(gamma->size());
}
VectorXd E1_dn_K(RDM1* gamma,int i){
    return VectorXd::Zero(gamma->size());
}
double E1_dn_Kn(RDM1* gamma,int i){
    return 0;
}
VectorXd E1_ddn_K(RDM1* gamma,int i, int j){
    return VectorXd::Zero(gamma->size());
}
double E1_ddn_Kn(RDM1* gamma, int i){
    return 0;
}