#include <math.h>
#include <eigen3/Eigen/Core>
#include <iostream>

#include "numerical_deriv.hpp"
#include "../classes/1RDM_class.hpp"
#include "../classes/Functional_class.hpp"

using namespace std;
using namespace Eigen;

#include<iomanip>
//Compute a numerical approximation of the gradient of the functional func at 1RDM gamma
//(used to test things)
VectorXd grad_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; VectorXd res (ll);
    if (do_n){
        for (int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            double coeff = 2.; if(gamma->n(i)> 2.-epsi || gamma->n(i)<epsi){coeff=1;}
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(coeff*epsi);
        }
    }
    if (do_no){
        for (int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m(gamma);
            gamma_p.l_theta(i-l) += epsi; gamma_m.l_theta(i-l) -= epsi;
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.segment(0,l);}
    if (do_no &&(not do_n)){return res.segment(l,ll-l);}
    else{return res;}
}
//Compute a numerical approximation of the gradient of the functional func at 1RDM gamma respect to the entries of U and the occupations.
//(used to test things)
VectorXd grad_func_aux(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int l2 = l*l; VectorXd res (l2+l);
    if (do_n){
        for (int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.set_n(i,gamma->n(i)+epsi); gamma_m.set_n(i,gamma->n(i)-epsi); 
            double coeff = 2.; if(gamma->n(i)> 2.-epsi || gamma->n(i)<epsi){coeff=1;}
            res(i) = (func->E(&gamma_p) - func->E(&gamma_m))/(coeff*epsi);
        }
    }
    if (do_no){
        int k= l;
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
                gamma_p.set_deviation(i,j,epsi); gamma_m.set_deviation(i,j,-epsi); 
                res(k) = (func->E(&gamma_p) - func->E(&gamma_m))/(2*epsi);
                k++;
            }
        }
    }
    if (do_n &&(not do_no)){return res.segment(0,l);}
    if (do_no &&(not do_n)){return res.segment(l,l2);}
    else{return res;}
}

//Compute a numerical approximation of the Hessian of the functional func at 1RDM gamma
//(used to test things, requires analytical gradient)
MatrixXd hess_func(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int ll = l*(l+1)/2; MatrixXd res =MatrixXd::Zero(ll,ll);
    if (do_n){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.x(i, gamma->x(i)+epsi) ; gamma_m.x(i, gamma->x(i)-epsi);
            gamma_p.solve_mu(); gamma_m.solve_mu();
            VectorXd grad_p = func->grad_E(&gamma_p,true,false); VectorXd grad_m = func->grad_E(&gamma_m,true,false);
            double coeff = 2.; if(gamma->n(i)> 2.-epsi || gamma->n(i)<epsi){coeff=1;}
            res.block(i,0,1,l) = (grad_p-grad_m).transpose()/(coeff*epsi);
        }
    }
    if (do_no){
        for(int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.l_theta(i-l) += epsi; gamma_m.l_theta(i-l) -= epsi;
            VectorXd grad_p = func->grad_E(&gamma_p,false,true); VectorXd grad_m = func->grad_E(&gamma_m,false,true);
            res.block(i,l,1,ll-l) = (grad_p-grad_m).transpose()/(2.*epsi);
        }
    }
    if(do_n && do_no){
        for(int i=l;i<ll;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.l_theta(i-l) += epsi; gamma_m.l_theta(i-l) -= epsi;
            VectorXd grad_p = func->grad_E(&gamma_p,true,false); VectorXd grad_m = func->grad_E(&gamma_m,true,false);
            res.block(i,0,1,l) = (grad_p-grad_m).transpose()/(2.*epsi);
            res.block(0,i,l,1) = (grad_p-grad_m)/(2.*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.block(0,0,l,l);}
    if (do_no &&(not do_n)){return res.block(l,l,ll-l,ll-l);}   
    else{return res;}
}

//Compute a numerical approximation of the Hessian of the functional func at 1RDM gamma respect to the entries of U and the occupations.
//(used to test things, requires analytical gradient)
MatrixXd hess_func_aux(Functional* func, RDM1* gamma, bool do_n, bool do_no, double epsi){
    int l = gamma->size(); int l2 = l*l; MatrixXd res =MatrixXd::Zero(l2+l,l2+l);
    if (do_n){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.set_n(i,gamma->n(i)+epsi); gamma_m.set_n(i,gamma->n(i)-epsi); 
            VectorXd grad_p = func->grad_aux(&gamma_p,true,false); VectorXd grad_m = func->grad_aux(&gamma_m,true,false);
            double coeff = 2.; if(gamma->n(i)> 2.-epsi || gamma->n(i)<epsi){coeff=1;}
            res.block(0,i,l,1) = (grad_p-grad_m)/(coeff*epsi);
        }
    }
    if (do_no){
        for (int i=0;i<l;i++){
            for (int j=0;j<l;j++){
                for (int p=0;p<l;p++){
                    for (int q=0;q<l;q++){
                        RDM1 gamma_pp (gamma); RDM1 gamma_p0 (gamma); RDM1 gamma_0p (gamma); 
                        RDM1 gamma_m0 (gamma); RDM1 gamma_0m (gamma); RDM1 gamma_mm (gamma);
                        gamma_pp.set_deviation(i,j,epsi); gamma_pp.set_deviation(p,q,epsi);
                        gamma_p0.set_deviation(i,j,epsi); gamma_0p.set_deviation(p,q,epsi); 
                        gamma_m0.set_deviation(i,j,-epsi); gamma_0m.set_deviation(p,q,-epsi);
                        gamma_mm.set_deviation(i,j,-epsi); gamma_mm.set_deviation(p,q,-epsi);
                        double Epp = func->E(&gamma_pp); double Ep0 = func->E(&gamma_p0); double E0p = func->E(&gamma_0p); 
                        double Em0 = func->E(&gamma_m0); double E0m = func->E(&gamma_0m); double Emm = func->E(&gamma_mm);
                        double E00 = func->E(gamma);
                        res((i+1)*l+j,(p+1)*l+q) = (Epp-Ep0-E0p+2.*E00-Em0-E0m+Emm)/(2.*epsi*epsi);
                    }
                }
            }
        }
    }
    if(do_n && do_no){
        for(int i=0;i<l;i++){
            RDM1 gamma_p (gamma); RDM1 gamma_m (gamma);
            gamma_p.set_n(i,gamma->n(i)+epsi); gamma_m.set_n(i,gamma->n(i)-epsi); 
            VectorXd grad_p = func->grad_aux(&gamma_p,false,true); VectorXd grad_m = func->grad_aux(&gamma_m,false,true);
            double coeff =2.; if(gamma->n(i)> 2.-epsi || gamma->n(i)<epsi){coeff=1;}
            res.block(i,l,1,l2) = (grad_p-grad_m).transpose()/(coeff*epsi);
            res.block(l,i,l2,1) = (grad_p-grad_m)/(coeff*epsi);
        }
    }
    if (do_n &&(not do_no)){return res.block(0,0,l,l);}
    if (do_no &&(not do_n)){return res.block(l,l,l2,l2);}   
    else{return res;}
}
