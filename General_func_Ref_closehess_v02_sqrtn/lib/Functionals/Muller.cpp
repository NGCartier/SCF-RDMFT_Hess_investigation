#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>

using namespace std;

#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"
#include "Muller.hpp"


VectorXd Muller_n_K(RDM1* gamma){
    return gamma->sqrtn();
}

VectorXd Muller_dn_K(RDM1* gamma,int i){
    return gamma->dsqrt_n(i); 
}

double Muller_dn_Kn(RDM1* gamma, int i){
    return 1.;
}

VectorXd Muller_ddn_K(RDM1* gamma, int i,int j){
    return gamma->ddsqrt_n(i,j);
}

double Muller_ddn_Kn(RDM1* gamma, int i){
    return 0.;
}