
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <limits>
#include <cmath>
#include <iostream>

#include "1RDM_class.hpp"


using namespace std;
using namespace Eigen;

const double precision = 0.1*sqrt(DBL_EPSILON);

void H_init(MatrixXd* hess_, VectorXd s, VectorXd y, int l){ 
    int ls = s.size();
    (*hess_) = MatrixXd::Zero(ls,ls);
    //if (l==0){
        double sigma = y.dot(s)/s.squaredNorm();
        for (int i=0;i<ls;i++){
            (*hess_)(i,i) = sigma;
        }
    /*}
    else{
        VectorXd y_n = y.segment(0,l); VectorXd s_n = s.segment(0,l); 
        double sigma_n = y_n.dot(s_n)/s_n.squaredNorm();
        for (int i=0;i<l;i++){
            (*hess_)(i,i) = sigma_n;
        }
        VectorXd y_no = y.segment(l,ls-l); VectorXd s_no = s.segment(l,ls-l); 
        double sigma_no = y_no.dot(s_no)/s_no.squaredNorm();
        for (int i=l;i<ls;i++){
            (*hess_)(i,i) = sigma_no;
        }
    }*/
}


void SR1(void* f_data){ 
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,step,y,data->gamma->size());}
    double sNorm = step.norm(); 
    VectorXd u = y -data->hess_*step; double uNorm = u.norm();
    double sigma = u.dot(step); 
    if (abs(sigma) >precision*sNorm*uNorm){
        data->hess_ += u*u.transpose()/sigma; 
    }
}

void BFGS(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,step,y,data->gamma->size());}
    double sNorm = step.norm(); double sTy = step.dot(y);
   
    VectorXd Hs = data->hess_*step;
    double sHs = step.dot(Hs);
    data->hess_ += y*y.transpose()/sTy - Hs*Hs.transpose()/sHs; 
}

void DFP(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,step,y,data->gamma->size());}
    double sNorm = step.norm(); double sTy = step.dot(y);
    MatrixXd u = MatrixXd::Identity(step.size(),step.size()) - y*step.transpose()/sTy;
    data->hess_ = u*data->hess_*u.transpose() + y*y.transpose()/sTy;
}

void SR1_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
     MatrixXd Jinv = Jt.completeOrthogonalDecomposition().pseudoInverse(); // /!\ inefficiant methode 
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,Jinv*step,Jinv*y,data->gamma->size());}
    double sNorm = step.norm(); 
    MatrixXd Htilde = Jt*data->hess_*J+data->func->ddE_known(data->gamma);
    VectorXd u = y -Htilde*step; double uNorm = u.norm();
    VectorXd v = Jinv*u; 
    double sigma = u.dot(step); 
    if(abs(sigma) >precision*sNorm*uNorm){
        data->hess_ += v*v.transpose()/sigma;
        data->update_hess = true;
    }
    else{
        data->update_hess = false;
    }
}

void BFGS_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = Jt.completeOrthogonalDecomposition().pseudoInverse(); // /!\ inefficiant methode   
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,Jinv*step,Jinv*y,data->gamma->size());}
    double sNorm = step.norm(); double sTy = step.dot(y);
    MatrixXd Htilde = Jt*data->hess_*J+data->func->ddE_known(data->gamma);
    VectorXd Hs = Htilde*step;
    VectorXd u = Jinv*y ; double uNorm = u.norm();
    VectorXd v = Jinv*Hs; double sHs = step.dot(Hs);
    data->hess_ += u*u.transpose()/sTy - v*v.transpose()/sHs;    
}

void DFP_aux(void* f_data){
    data_struct *data = (data_struct*) f_data; 
    if(data->niter==0){ return ;}
    VectorXd step = data->x2 - data->x1; VectorXd y = data->grad2 - data->grad1;
    MatrixXd J = data->func->Jac(data->gamma); MatrixXd Jt = J.transpose();
    MatrixXd Jinv = Jt.completeOrthogonalDecomposition().pseudoInverse(); // /!\ inefficiant methode  
    if(data->niter==1 && data->do_1st_iter){H_init(&data->hess_,Jinv*step,Jinv*y,data->gamma->size());}
    double sNorm = step.norm(); double sTy = step.dot(y);
    MatrixXd B = Jt*data->hess_*J;
    MatrixXd u = Jinv*(MatrixXd::Identity(step.size(),step.size())-y*step.transpose()/sTy);
    VectorXd v = Jinv*y;
    data->hess_ = u*B*u.transpose() + v*v.transpose()/sTy;
}