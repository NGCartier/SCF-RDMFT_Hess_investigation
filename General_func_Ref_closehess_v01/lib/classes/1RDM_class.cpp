#include <stdio.h>
#include <math.h>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <float.h>
#include <iostream>
#include <fstream>
#include <iomanip>      
#include <chrono>
#include <tuple>
#include <nlopt.hpp> 
#include <vector>

#include "1RDM_class.hpp"
#include "Functional_class.hpp"
#include "../numerical_deriv/numerical_deriv.hpp"
#include "EBI_add.hpp"

#include "../../fides-cpp-main/src/fides.cpp"
#include "../../fides-cpp-main/src/hessian_approximation.cpp"
#include "../../fides-cpp-main/src/minimize.cpp"
#include "../../fides-cpp-main/src/stepback.cpp"
#include "../../fides-cpp-main/src/steps.cpp"
#include "../../fides-cpp-main/src/subproblem.cpp"
#include "../../fides-cpp-main/src/trust_region.cpp"

using namespace std;
using namespace Eigen;

/*Constructor for the 1RDM
\param result the empty 1RDM
*/
RDM1::RDM1(){
    MatrixXd null (0,0); 
    n_elec = 0;
    E_nuc  = 0.;
    ovlp   = null;
    int1e  = null;
    int2e  = null;
    int2e_x = null;
    vector<int> v(1,0); 
    omega.push_back(v);
    mu = VectorXd(0);
    V_ = VectorXd(0);
    W_ = VectorXd(0);
    x_ = VectorXd(0); 
    l_theta = VectorXd(0);
    no_ = null;
    computed_V_ = VectorXi(0);  
    computed_W_ = VectorXi(0);
    deviation_ = MatrixXd(0,0);
}
/*Constructor for the 1RDM
\param args ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;
    int l= overlap.rows();
    vector<int> v(l); iota(v.begin(), v.end(), 0);
    omega.push_back(v);
    mu = VectorXd::Constant(1,0.);
    V_ = VectorXd::Constant(1,0.);
    W_ = VectorXd::Constant(1,0.);
    computed_V_ = VectorXi::Constant(1,false);
    computed_W_ = VectorXi::Constant(1,false);
    l_theta = VectorXd::Zero(l*(l-1)/2);
    deviation_ = MatrixXd::Zero(l,l);
    no_ = overlap.inverse().sqrt();
    x_.resize(overlap.rows()); 
    if (ne>l){
        for (int i= 0;i<l; i++){
            if (i>ne){ set_n(i,sqrt(2.)); }
            else {set_n(i,1.);} 
        }
    }
    else {
        for (int i= 0;i<ne; i++){
            set_n(i,1.);
        } 
    } 
}
/*Constructor for the 1RDM
\param args n: vector of the sqrt of the occupations
            no: matrix of the NOs
            ne: number of electron
            Enuc: nuclei energy 
            overlap: overlap matrix
            elec1int: 1electron integrals matrix
            elec2int: 2electrons intergrals matrix
            exch2int: 2electrons intergrals matrix permutated to give the exchange energy
\param result the corresponding 1RDM
*/
RDM1::RDM1(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int, MatrixXd exch2int){
    
    n_elec = ne;
    E_nuc = Enuc;
    ovlp = overlap;
    int1e = elec1int;
    int2e = elec2int;
    int2e_x = exch2int;
    int l = occ.size();
    vector<int> v(l); iota(v.begin(), v.end(), 0);
    omega.push_back(v);
    deviation_ = MatrixXd::Zero(l,l);
    mu = VectorXd::Constant(1,0.);
    V_ = VectorXd::Constant(1,0.);
    W_ = VectorXd::Constant(1,0.);
    computed_V_ = VectorXi::Constant(1,false);
    computed_W_ = VectorXi::Constant(1,false);
    x_.resize(occ.size());
    set_n(occ); //Initialise x
    l_theta = VectorXd::Zero(l*(l-1)/2);
    no_ = orbital_mat;
}
/*Copy a instance of the RDM1 class*/
RDM1::RDM1(const RDM1* gamma){
    n_elec = gamma->n_elec;
    E_nuc  = gamma->E_nuc;
    ovlp   = gamma->ovlp;
    int1e  = gamma->int1e;
    int2e  = gamma->int2e;
    int2e_x = gamma->int2e_x;
    deviation_ = gamma->deviation_;
    x_  = gamma->x_;
    l_theta = gamma->l_theta;
    no_ = gamma->no_;
    mu  = gamma->mu;
    V_  = gamma->V_;
    W_ = gamma->W_;
    computed_V_ = gamma->computed_V_;
    computed_W_ = gamma->computed_W_;
    omega = gamma->omega;   
}
RDM1::~RDM1(){}; 

/* Affectation to x and its elements 
(needs to be a method to update computed_V_) */
void RDM1::x(int i, double xi){
    x_(i) = xi;
    int g = find_subspace(i);
    computed_V_(g) = false;
}

void RDM1::x(VectorXd x0){
    x_ = x0;
    computed_V_ = VectorXi::Constant(omega.size(),false);
    computed_W_ = VectorXi::Constant(omega.size(),false);
}

/*Get the index of the subspace containing index i*/
int RDM1::find_subspace(int i) const{
    if(omega.size()==1){ //avoid useless loop if no subspaces
        return 0;
    }
    else{
        int g=-1; int j=0;
        while(g==-1){
            if (find(omega[j].begin(), omega[j].end(), i) != omega[j].end()){
                g=j;
            }
            j++;
        }
        return g;
    }
}

/*Get the ith occupation from x and mu */
double RDM1::n(int i) const{
    int g = find_subspace(i); 
    return erf(x(i)+mu(g))+1.;
}
/*Get all the occupations from x and mu */
VectorXd RDM1::n() const{
    VectorXd res(size());
    for (int i=0;i<size();i++){
        res(i) = n(i);
    }
    return res;
}
/*Compute x_i cooresponding to the given occupation*/ 
void RDM1::set_n(int i,double ni){
    int g = find_subspace(i); 
    x(i, erfinv(ni-1.)-mu(g));
}
/*Compute x corresponding to the given occupations*/
void RDM1::set_n(VectorXd n){
    for (int i=0;i<n.size();i++){
        set_n(i,n(i));
    }
}
/*Compute V (used in computation of derivatives)*/
double RDM1::get_V(int g){
    if (computed_V_(g)){
        return V_(g);
    }
    else{
        V_(g) = 0; 
        for (int i:omega[g]){
            V_(g) += derf(x(i)+mu(g));
        }
        if (V_(g)==0.){ //avoid numerical issues
            V_(g)= 1e-15; 
        }
        computed_V_(g) = true;
        return V_(g);
    }
}

/*Compute W */
double RDM1::get_W(int g){
    if (computed_W_(g)){
        return W_(g);
    }
    else{
        W_(g) = 0;
        for (int i:omega[g]){
            W_(g) += dderf(x(i)+mu(g));
        } 
        computed_W_(g) = true;
        return W_(g);
    }
}

/*Computes the matrix form of the 1RDM*/
MatrixXd RDM1::mat() const { 
    MatrixXd N = n().asDiagonal();
    MatrixXd NO = no();
    return NO*N*NO.transpose();
}
/* Computes USNST */
MatrixXd RDM1::matL() const {
    MatrixXd N = n().asDiagonal();
    MatrixXd NO = no();
    MatrixXd S = no_.transpose();
    return NO*N*S;
}
/* Computes SNSTUT */
MatrixXd RDM1::matR() const {
    MatrixXd N = n().asDiagonal();
    MatrixXd NO = no().transpose();
    MatrixXd S = no_;
    return S*N*NO;
}


/* Derivative of the 1RDM respect to its ith parameters */
MatrixXd RDM1::dmat(int i){
    int l = size(); 
    if(i<l){
        MatrixXd NO = no();
        return NO*dn(i).asDiagonal()*NO.transpose();
    }
    else{
        MatrixXd N = n().asDiagonal();
        MatrixXd NO = no(); MatrixXd dNO = dno(i-l);
        return dNO*N*NO.transpose() + NO*N*dNO.transpose(); 
    }
    
}


/* Derivative of ith occupation respect to the parameters jth parameter x_j */
double RDM1::dn(int i, int j){
    int f = find_subspace(i);
    int g = find_subspace(j);
    if (f==g){
        double dn_x = derf(x(i)+mu(f));
        double dn_mu = -derf(x(j)+mu(f))*dn_x/get_V(f);
        return (i==j)*dn_x+dn_mu;
    }
    else{
        return 0.;
    }
}
/* Derivative of the occupations respect to the parameters ith parameter x_i */
VectorXd RDM1::dn(int i){
    int l = size(); VectorXd res(l);
    for (int j=0;j<l;j++){
        res(j) = dn(j,i);
    }
    return res;
}
/* 2nd derivative of kth occupation respect to the ith and jth parameters */
double RDM1::ddn(int k,int i,int j){
    int f  = find_subspace(k); 
    int g1 = find_subspace(i);
    int g2 = find_subspace(j);
    if (f==g1 && f==g2){
        double derf_i = derf(x(i)+mu(f));
        double derf_j = derf(x(j)+mu(f));
        double derf_k = derf(x(k)+mu(f));
        double dmu_i = -derf_i/get_V(f);
        double dmu_j = -derf_j/get_V(f);
        double dderf_i = dderf(x(i)+mu(f));
        double dderf_j = dderf(x(j)+mu(f));
        double dderf_k = dderf(x(k)+mu(f));
        double ddmu  = -(get_W(f)*dmu_i*dmu_j+dderf_i*dmu_j+dderf_j*dmu_i+(i==j)*dderf_i)/get_V(f);
        return dderf_k * ((i==k)+dmu_i) * ((j==k)+dmu_j) + derf_k*ddmu;
    }
    else{
        return 0;
    }
}

/* 2nd derivative of the occupations respect to the ith and jth parameter */
VectorXd RDM1::ddn(int i, int j){
    int l = size(); VectorXd res(l);
    for (int k=0;k<l;k++){
        res(k) = ddn(k,i,j);
    }
    return res;
}

/* Derivative of the square root of ith occupation respect to the parameters jth parameter x_j */
double RDM1::dsqrt_n(int i,int j){
    int f = find_subspace(i);
    int g = find_subspace(j);
    if(f==g){
        double derf_x = derf(x(i)+mu(f));
        double dn_x = derf_x/(2.*sqrt( erf(x(i)+mu(f))+1));
        double dn_mu = -derf(x(j)+mu(f))*dn_x/get_V(f);
        return (i==j)*dn_x+dn_mu;
    }
    else{
        return 0;
    }
}

/* Derivative of the square root of the occupations respect to the parameters ith parameter x_i */
VectorXd RDM1::dsqrt_n(int i){
    int l = size(); VectorXd res(l);
    for (int j=0;j<l;j++){
        res(j) = dsqrt_n(j,i);
    }
    return res;
}

/* 2nd derivative of the square root of kth occupation respect to the ith and jth parameters */
double RDM1::ddsqrt_n(int k, int i,int j){
    return ddn(k,i,j)/sqrt(4.*n(k))- dn(k,i)*dn(k,j)/(4.*pow(n(k),3./2.));
}
/* 2nd derivative of the square root of the occupations respect to the ith and jth parameters */
VectorXd RDM1::ddsqrt_n(int i,int j){
    int l = size(); VectorXd res(l);
    for (int k=0;k<l;k++){
        res(k) = ddsqrt_n(k,i,j);
    }
    return res;
}

/* Compute the value of mu (shared paramerter of EBI representation) from x */
void RDM1::solve_mu(){
    if(omega.size()==1){
        solve_mu_aux(this); // see EBI_add.cpp
    }
    else{
        solve_mu_subs_aux(this); // see EBI_add.cpp
    }
    computed_V_ = VectorXi::Constant(omega.size(),false);
    computed_W_ = VectorXi::Constant(omega.size(),false);
}
/* Return a boolean vector of size l initialised at false */
Vector<bool,Eigen::Dynamic> init_V(int l){
    Vector<bool,Eigen::Dynamic> v(l);
    for (int i=0;i<l;i++){
        v(i) = false;
    }
    return v;
}

/* Current value of the NO matrix */
MatrixXd RDM1::no() const{
    //return Givens(no_,&l_theta,i); //faster but incompatible with numerical approximation of the gradient/hessian
    int l = size();
    return no_*(Givens(MatrixXd::Identity(l,l),&l_theta) + deviation_);
}

/* Current derivative of the NO matrix */
MatrixXd RDM1::dno(int i) const{
    //return dGivens(no_,&l_theta,i);
    int l = size();
    return no_*(dGivens(MatrixXd::Identity(l,l),&l_theta,i) + deviation_);
}

/*Prints the time lapse between t0 and t1 divided by iter (default iter=1)*/
void print_t(chrono::high_resolution_clock::time_point t1, chrono::high_resolution_clock::time_point t0, int iter){
    auto t_nano = chrono::duration_cast<chrono::nanoseconds>(t1-t0).count();
    t_nano /= iter;
    if(t_nano<1000){
        cout<<t_nano<<"ns";
        return;
    }
    auto t_micro = chrono::duration_cast<chrono::microseconds>(t1-t0).count();
    t_micro /= iter;
    t_nano -= t_micro*1000;
    if(t_micro<1000){
        cout<<t_micro<<"µs "<<t_nano<<"ns";
        return;
    }
    auto t_milli = chrono::duration_cast<chrono::milliseconds>(t1-t0).count();
    t_milli /= iter;
    t_micro -= t_milli*1000;
    if(t_milli<1000){
        cout<<t_milli<<"ms "<<t_micro<<"µs";
        return;
    }
    auto t_sec = chrono::duration_cast<chrono::seconds>(t1-t0).count();
    t_sec /= iter;
    t_milli -= t_sec*1000;
    if(t_sec<60){
        cout<<t_sec<<"s "<<t_milli<<"ms";
        return;
    }
    auto t_min = chrono::duration_cast<chrono::minutes>(t1-t0).count();
    t_min /= iter;
    t_sec -= t_min*60;
    if(t_min<60){
        cout<<t_min<<"'"<<t_sec<<"s";
        return;
    }
    auto t_hour = chrono::duration_cast<chrono::hours>(t1-t0).count();
    t_hour /= iter;
    t_min -= t_hour*60;
    cout<<t_hour<<"h"<<t_min<<"'";
}

VectorXd negative_eigvls(MatrixXd M,double epsi){
    VectorXd eigs  = M.selfadjointView<Upper>().eigenvalues();
    vector<double>neigvls; 
    for (int i=0;i<eigs.size();i++){
        if(eigs(i)<-epsi){
            neigvls.push_back(eigs(i));
        }
    }
    VectorXd res = VectorXd::Map(neigvls.data(),neigvls.size());
    return res;
}
/*Optimises the occupations (n) and NOs (no) of the 1RDM with respect to the energy minimisation
\param args func: the functional to use
            disp: if >1 displais details about the computation
            epsi: relative precision required for the optimisation
            maxiter: maximum number of iterations for one optimisation of the NOs/occupations, and maximum number of 
                     calls to those optimisations
the occupations and NOs are optimised in-place
*/
void RDM1::opti(Functional* func, int disp, double epsi, int maxiter, string hess_approx, string file, string cond){
    ofstream ofile; ofile.open(file+".txt");
    ofile<<setprecision(8); cout<<setprecision(8);
    ofstream ofile_behav; ofile_behav.open(file+"_behaviour.txt");
    ofile_behav<<setprecision(8); 
    
    int kmax = max(100,maxiter);
    auto t_init = chrono::high_resolution_clock::now();
    int l = size(); int nit= 0; int ll = l*(l-1)/2; int k =0;
    double E = func->E(this); double E_bis = DBL_MAX;
    
    bool detailed_disp;
    if (disp>2){detailed_disp = true;}
    else {detailed_disp = false;}
    
    auto t0 = chrono::high_resolution_clock::now(); int nit_i=0;
    while( ( abs((E_bis-E)/E)>epsi ) && k<kmax){ 
        E_bis = E;
        auto res  = opti_aux(this, func, hess_approx, &ofile, &ofile_behav, cond, sqrt(epsi)*0.1, detailed_disp, maxiter);
        int nit_i = get<1>(res); E = get<0>(res);    
        nit += nit_i; 
        k++;
        if(disp>1){
            cout<<"iter "<<k<<" E="<<E<<" |grad|="<<func->grad_E(this).norm()<<" # iter="<<nit_i<<endl;
        }
        ofile<<"---------"<<endl;
    }
    ofile.close();
    auto t_fin = chrono::high_resolution_clock::now();
    if (disp>0){ 
        cout<<endl;
        cout<<"Computational time "; print_t(t_fin,t_init); cout<<" total # of iter "<<nit<<endl;
    }
}

//Default objective function for the optimistion called by the minimizer
fides::cost_fun_ret_t f_closehess(DynamicVector<double> X, void* f_data){
    int ll = X.size(); int l = (sqrt(8*ll+1)-1)/2; int l2 =l*l;
    VectorXd x = VectorXd::Map(X.data(),ll); 
    data_struct *data = (data_struct*) f_data;
    MatrixXd Htemp = data->func->hess_E(data->gamma,&data->hess_); //old Hessian
    //Update old step and y for the auxiliary Hessian
    data->x1 = data->x2; data->grad1 = data->grad2;
    //Update gamma
    data->gamma->x(x.segment(0,l));
    data->gamma->solve_mu();
    data->gamma->l_theta = x.segment(l,ll-l);
    MatrixXd W_J = data->func->compute_WJ(data->gamma);
    MatrixXd W_K = data->func->compute_WK(data->gamma);
    //Update new step and y
    data->x2 = x; data->grad2 = data->func->grad_E(data->gamma);
    MatrixXd H_analy = data->func->hess_aux(data->gamma);
    MatrixXd Hdiff = H_analy-data->Hanaly; data->Hanaly = H_analy; 
    //Update E grad and Hess 
    double E = data->func->E(data->gamma, &W_J, &W_K);

    if(data->hess_approx == "exa" || (data->niter%data->interval==0 && data->niter !=0)){ data->hess_ = H_analy; }
    else if(data->hess_approx == "BFGS"){ BFGS_aux(f_data); }
    else if(data->hess_approx == "SR1"){ SR1_aux(f_data);}
    
    if(data->exa_nn){
        data->hess_.block(0,0,l,l) = H_analy.block(0,0,l,l);
    }
    if(data->exa_NONO){
        data->hess_.block(l,l,l2,l2) = H_analy.block(l,l,l2,l2);
    }
    if(data->exa_nNO){
        data->hess_.block(0,l,l,l2) = H_analy.block(0,l,l,l2);
        data->hess_.block(l,0,l2,l) = H_analy.block(l,0,l2,l);
    }
    
    if (data->update_hess){ Htemp = data->func->hess_E(data->gamma,&data->hess_);} // If valid update Hessian
    DynamicVector<double> grad (ll); DynamicMatrix<double> hess (ll,ll); 
    for (int i=0;i<ll;i++){
        grad[i] = data->grad2(i);
        for (int j =0;j<=i;j++){
            hess(i,j) = Htemp(i,j); hess(j,i) = Htemp(i,j); 
        }
        
    }
    data->niter++;

    //Returns/writes results
    MatrixXd H_error = data->hess_ - H_analy; double Hnorm = H_analy.norm();
    *data->ofile<<"E="<<E<<" |H_er|="<<H_error.norm()<<" |H_er_nn|="<<H_error.block(0,0,l,l).norm()\
    <<" |H_er_NONO|="<<H_error.block(l,l,l2,l2).norm()<<" |H_er_nNO|="<<sqrt(2.)*H_error.block(0,l,l,l2).norm()<<endl;

    if (data->disp){ //wether we want to output the negative eigenvalues
        cout<<"H app aux eigvals=";negative_eigvls(data->hess_);cout<<endl;
        cout<<"H exa aux eigvals=";negative_eigvls(H_analy);cout<<endl;
        cout<<"H approx  eigvals=";negative_eigvls(Htemp);cout<<endl;
        cout<<"-------------------"<<endl;
    }

    //Used to get additional infos for convergence 
    *data->ofile_behav<<"E="<<E<<" #negative lambda="<<negative_eigvls(data->hess_).size()<<" |Hexa_k-Hexa_{k-1}|="<<(Hdiff).norm()\
    <<" |s|="<<(data->x2-data->x1).norm()<<" |grad|="<<data->grad2.norm()<<endl;
    return make_tuple(E,grad,hess);
}

fides::cost_fun_ret_t f_generalhess(DynamicVector<double> X, void* f_data){
    int ll = X.size(); int l = (sqrt(8*ll+1)-1)/2; 
    VectorXd x = VectorXd::Map(X.data(),ll); 
    data_struct *data = (data_struct*) f_data;
    //Update old step and y for the auxiliary Hessian
    data->x1 = data->x2; data->grad1 = data->grad2;
    //Update gamma
    data->gamma->x(x.segment(0,l));
    data->gamma->solve_mu();
    data->gamma->l_theta = x.segment(l,ll-l);
    MatrixXd W_J = data->func->compute_WJ(data->gamma);
    MatrixXd W_K = data->func->compute_WK(data->gamma);
    //Update new step and y
    data->x2 = x; data->grad2 = data->func->grad_E(data->gamma);
    MatrixXd hess_aux = data->func->hess_aux(data->gamma);
    MatrixXd H_analy = data->func->hess_E(data->gamma,&hess_aux);
    MatrixXd Hdiff = H_analy-data->Hanaly; data->Hanaly = H_analy; 
    //Update E grad and Hess
    double E = data->func->E(data->gamma, &W_J, &W_K);

    if(data->hess_approx == "exa" || (data->niter%data->interval==0 && data->niter !=0)){ data->hess_ = H_analy; }
    else if(data->hess_approx == "BFGS"){ BFGS(f_data); }
    else if(data->hess_approx == "SR1"){ SR1(f_data);}
    
    if(data->exa_nn){
        for(int i=0;i<l;i++){
            for(int j=0;j<=i;j++){
                data->hess_(i,j) = H_analy(i,j); data->hess_(j,i) = H_analy(j,i);
            }
        }
    }
    if(data->exa_NONO){
        for(int i=l;i<ll;i++){
            for(int j=l;j<=i;j++){
                data->hess_(i,j) = H_analy(i,j); data->hess_(j,i) = H_analy(j,i);
            }
        }
    }
    if(data->exa_nNO){
        for(int i=l;i<ll;i++){
            for(int j=0;j<l;j++){
                data->hess_(i,j) = H_analy(i,j); data->hess_(j,i) = H_analy(j,i);
            }
        }
    }
    DynamicVector<double> grad (ll); DynamicMatrix<double> hess (ll,ll); 
    for (int i=0;i<ll;i++){
        grad[i] = data->grad2(i);
        for (int j =0;j<=i;j++){
            hess(i,j) = data->hess_(i,j); hess(j,i) = data->hess_(i,j); 
        }
    }
    data->niter++; 

    //Returns/writes results
    MatrixXd H_error = data->hess_ - H_analy; double Hnorm = H_analy.norm();
    *data->ofile<<"E="<<E<<" |H_er|="<<H_error.norm()<<" |H_er_nn|="<<H_error.block(0,0,l,l).norm()\
    <<" |H_er_NONO|="<<H_error.block(l,l,ll-l,ll-l).norm()<<" |H_er_nNO|="<<sqrt(2.)*H_error.block(0,l,l,ll-l).norm()<<endl;
    
    if (data->disp){ //wether we want to output the negative eigenvalues
        cout<<"H approx eigvals="<<negative_eigvls(data->hess_).transpose()<<endl;
        cout<<"H exact  eigvals="<<negative_eigvls(H_analy).transpose()<<endl;
    }

    //Used to get additional infos for convergence 
    *data->ofile_behav<<"E="<<E<<" #negative lambda="<<negative_eigvls(data->hess_).size()<<" |Hexa_k-Hexa_{k-1}|="<<(Hdiff).norm()\
    <<" |s|="<<(data->x2-data->x1).norm()<<" |grad|="<<data->grad2.norm()<<endl;
    return make_tuple(E,grad,hess);
}

/* Optimises the occupatiosn and orbitals of the 1RDM with respect to the minimisation of the energy
\param args gamma: 1RDM
            func: functional
            epsilon: required precision
            eta: acceptance on the constraint violation
            disp: get detail on the optimisation /TO DO/
            maxiter: maximum number of iterations
the occupations are optimised in-place
\param results  corresponding energy, number of iterations
*/
tuple<double,int> opti_aux(RDM1* gamma, Functional* func, string hess_approx, ofstream* ofile, ofstream* ofile_behav, 
                            string cond, double epsilon, bool disp, int maxiter){
    int l = gamma->size(); int ll = l*(l+1)/2; int l2 = l*l;
    DynamicVector<double>  x(ll); DynamicVector<double> lb (ll); DynamicVector<double> ub (ll);
    for (int i = 0; i < l; i++) {
        x[i] = gamma->x(i); lb[i] = -4.; ub[i] = 4.;
    }
    for (int i=l; i<ll;i++){
        x[i] = gamma->l_theta(i-l); lb[i] = -M_PI; ub[i] = M_PI;
    }
    data_struct f_data;
    f_data.gamma = gamma; f_data.func = func; 
    f_data.x2.resize(ll); f_data.x2.segment(0,l) = gamma->x();/*0?*/ f_data.x2.segment(l,ll-l) = gamma->l_theta;  
    f_data.grad2 = func->grad_E(gamma); f_data.disp = disp;
    f_data.niter=0; f_data.ofile = ofile; f_data.ofile_behav = ofile_behav; f_data.do_1st_iter = true; f_data.update_hess = true;
    f_data.exa_nn = false; f_data.exa_NONO = false; f_data.exa_nNO = false; f_data.interval = INT_MAX;
    
    fides::Options options; options.maxiter = maxiter; options.refine_stepback = false; 
     
    options.fatol = 1e-17; options.frtol = 1e-17; //convergence in f usually not sufficiant to ensure convergence
    if(cond=="xtol" || cond=="stol" || cond=="x" || cond=="s"){
        options.xtol = epsilon; options.gatol = 1e-17; options.grtol = 1e-17;
    }
    else if (cond=="gtol" || cond=="gatol" || cond=="g"){
        options.gatol = epsilon; options.xtol = 1e-17; options.grtol = 1e-17;
    }
    else if (cond=="grtol"){
        options.grtol = epsilon; options.gatol = 1e-17; options.xtol = 1e-17;
    }

    //Determines the Hessian to use depending on hess_apporox
    fides::cost_function_t f_; 
    if (hess_approx.find("_nn") != string::npos){
        //Will replace the occupation-occupation block of the Hessian by the exact one
        f_data.exa_nn = true;
    }
    if (hess_approx.find("_NONO") != string::npos){
        //Will replace the NO-NO block of the Hessian by the exact one
        f_data.exa_NONO = true;
    }
    if (hess_approx.find("_nNO") != string::npos){
        //Will replace the occupation-NO blocks of the Hessian by the exact one
        f_data.exa_nNO = true;
    }
    if (hess_approx.find("exa") != string::npos && hess_approx.find("start") == string::npos){
        //Use the exact Hessian
        f_data.hess_approx = "exa";
    }
    else if (hess_approx.find("SR1") != string::npos){
        //Use SR1 approximation of the Hessian
        f_data.hess_approx = "SR1";
    }
    else if (hess_approx.find("BFGS") != string::npos){
        //Use BFGS approximation of the Hessian
        f_data.hess_approx = "BFGS";
    } 
    if (hess_approx.find("_aux") != string::npos){
        //Use an approximation of the auxiliary Hessian
        f_ = f_closehess; f_data.Hanaly = func->hess_aux(gamma);
        if(hess_approx.find("_exastart") != string::npos){
            //Use the exact Hessian at 1st iteration
            f_data.hess_ = func->hess_aux(gamma); f_data.do_1st_iter = true;
        } else{ f_data.hess_ = MatrixXd::Identity(l2+l,l2+l); }
    }
    //Note: if semothing else is specified no update of the Hessian will be called i.e. steepest descent.
    else{
        //Use an approximation of the full Hessian
        MatrixXd Hexa = func->hess_aux(gamma);
        f_ = f_generalhess; f_data.Hanaly = func->hess_E(gamma,&Hexa);
        if(hess_approx.find("_exastart") != string::npos){
            //Use the exact Hessian at 1st iteration
            MatrixXd Hexa = func->hess_aux(gamma); f_data.hess_ = func->hess_E(gamma,&Hexa); 
            f_data.do_1st_iter = false;
        } else{ f_data.hess_ = MatrixXd::Identity(ll,ll); }
    }
    string::size_type p = hess_approx.find("hybrid");
    if(p != string::npos){
        //Use exact Hessian every M iterations
        string::size_type p2 = hess_approx.find('_',p+8);
        string m = hess_approx.substr(p+7,p2-p-1); int M;
        if (m.empty()){ M = 10; }
        else{ M = stoi(m); }
        f_data.interval = M; 
    }


    fides::Optimizer opti (f_,lb,ub,options,nullptr,&f_data);
    opti.minimize(x);
    return  make_tuple(opti.fval_, f_data.niter);
}

// Computes the unitary transformation of M parametrised by givens rotations of angles l_theta
MatrixXd Givens(MatrixXd M, const VectorXd* l_theta){
    int l = (sqrt(8*l_theta->size()+1)+1)/2; int k =0;
    for(int i=0;i<l;i++){
        for(int j=0;j<i;j++){
            double c = cos(l_theta->coeff(k)); double s = sin(l_theta->coeff(k));
            MatrixXd M_col_i = M.col(i); MatrixXd M_col_j = M.col(j);
            M.col(i) = c*M_col_i - s*M_col_j;
            M.col(j) = c*M_col_j + s*M_col_i;
            k++;
        }
    }
    return M;
}

// Computes pth derivative of a Givens transformation of M
MatrixXd dGivens(MatrixXd M, const VectorXd* l_theta,int p){
    int ll = l_theta->size(); int l = (sqrt(8*ll+1)+1)/2; int k =0;
    if(p>=ll){
        cout<<"Error : Invalide index"<<endl;
        return MatrixXd::Zero(0,0);
    }
    for(int i=0;i<l;i++){
        for(int j=0;j<i;j++){
            double c = cos(l_theta->coeff(k)); double s = sin(l_theta->coeff(k));
            MatrixXd M_col_i = M.col(i); MatrixXd M_col_j = M.col(j);
            if(k==p){
                M = MatrixXd::Zero(l,l);
                M.col(i) = -s*M_col_i - c*M_col_j;
                M.col(j) = -s*M_col_j + c*M_col_i;
            }
            else{
                M.col(i) =  c*M_col_i - s*M_col_j;
                M.col(j) =  c*M_col_j + s*M_col_i;
            }
            k++;
        }
        
    }
    return M;
}

// Computes 2nd pth-qth derivative of a Givens transformation of M
MatrixXd ddGivens(MatrixXd M, const VectorXd* l_theta,int p,int q){
    int ll = l_theta->size(); int l = (sqrt(8*ll+1)+1)/2; int k =0;
    if(p>=ll || q>=ll){
        cout<<"Error : Invalide index"<<endl;
        return MatrixXd::Zero(0,0);
    }
    for(int i=0;i<l;i++){
        for(int j=0;j<i;j++){
            double c = cos(l_theta->coeff(k)); double s = sin(l_theta->coeff(k));
            MatrixXd M_col_i = M.col(i); MatrixXd M_col_j = M.col(j);
            if(k==p || k==q){
                if (p==q){
                    M = MatrixXd::Zero(l,l);
                    M.col(i) = -c*M_col_i + s*M_col_j;
                    M.col(j) = -c*M_col_j - s*M_col_i;
                }
                else{
                    M = MatrixXd::Zero(l,l);
                    M.col(i) = -s*M_col_i - c*M_col_j;
                    M.col(j) = -s*M_col_j + c*M_col_i;
                }
            }
            else{
                M.col(i) =  c*M_col_i - s*M_col_j;
                M.col(j) =  c*M_col_j + s*M_col_i;
            }
            k++;
        }
        
    }
    return M;
}
