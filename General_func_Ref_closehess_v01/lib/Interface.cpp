#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
//Eigen3 & NLopt
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <nlopt.hpp>
//Types
#include <float.h>
#include <vector>
#include <string>
#include <tuple> 
//Print 
#include <iostream>
#include <fstream>
#include <iomanip>      
#include <chrono>

using namespace std;
using namespace Eigen;

#include "classes/1RDM_class.hpp"
#include "classes/Functional_class.hpp"
#include "classes/Matrix_Tensor_converter.cpp"
#include "numerical_deriv/numerical_deriv.hpp"
#include "classes/EBI_add.hpp"
#include "Interface.hpp"

#include "Functionals/HF.hpp"
#include "Functionals/Muller.hpp"

//Dictionary of functionals 
map<string, Functional> Funcs = {{"E1",E1_func},{"Hartree",Hartree_func},{"HF",HF_func},{"Muller",Muller_func}};


// Wrapper to python code
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;
PYBIND11_MODULE(Compute_1RDM, m){
    m.def("Optimize_1RDM", &Optimize_1RDM, " Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a functional returns the \
    optimized 1RDM (ie corresponding occupation and natural orbital matrix) and prints its energy");
    m.def("E",&E, "Given occupations, natural orbital matrix, number of electrons, \
    nuclear energy, overlap matrix, tensors of integrals (one and two bodies) and a \
    functional returns the corresponding energy");
    m.def("test",&test);
    m.doc() = "C++ extension to compute 1RDMs from PySCF";
}


//Used to test the library from Python
void test(VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int,string func){
                                
    int l = overlap.rows(); int l2 = pow(l,2); double epsi =1e-5;
    // (10)(32) permutation already presente, due to type conversion
    Tensor<double,4> T(l,l,l,l); T = TensorCast(elec2int,l,l,l,l);  Eigen::array<int,4> index({ 1,3,0,2 });
    Tensor<double,4> T2 = T.shuffle(index);
    MatrixXd elec2int_x(l2, l2); elec2int_x = MatrixCast(T2, l2, l2);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x);

    auto t0 = chrono::high_resolution_clock::now(); 
    for (int i = 0;i<1;i++){
    //code to test 
        
    cout.precision(6);
    int ll = l*(l-1)/2;

    
    VectorXd ltheta = VectorXd::Random(ll); 
    
    auto functional = Funcs.find(func);

    cout<<"ltheta="<<ltheta.transpose()<<endl;
    cout<<"E ="<<functional->second.E(&gamma)<<endl; 
    gamma.l_theta = ltheta;

    cout<<grad_func_aux(&functional->second,&gamma).transpose()<<endl;
    cout<<functional->second.grad_aux(&gamma).transpose()<<endl<<endl;

    MatrixXd H_aux = functional->second.hess_aux(&gamma);

    MatrixXd H_num = hess_func(&functional->second,&gamma);
    MatrixXd H_analy = functional->second.hess_E(&gamma,&H_aux);
    
    cout<<H_num.block(0,0,l,l)<<endl<<endl<<H_analy.block(0,0,l,l)<<endl;
    cout<<(H_num.block(0,0,l,l)-H_analy.block(0,0,l,l)).norm()<<endl<<endl;
    cout<<H_num.block(l,l,ll,ll)<<endl<<endl<<H_analy.block(l,l,ll,ll)<<endl;
    cout<<(H_num.block(l,l,ll,ll)-H_analy.block(l,l,ll,ll)).norm()<<endl<<endl;
    cout<<H_num.block(0,l,l,ll)<<endl<<endl<<H_analy.block(0,l,l,ll)<<endl;
    cout<<(H_num.block(0,l,l,ll)-H_analy.block(0,l,l,ll)).norm()<<endl<<endl;
    
    auto t1 = chrono::high_resolution_clock::now();
    print_t(t1,t0,1); cout<<endl;
    }   
}

/* 
\param arg    func: functional to use (modify this function to add new functinals)
              disp: if <1 prints details of the computation 
              epsi: relative error required 
              Maxiter: maximum number of iteration for one optimisation of the occupations/NOs
              other arguments are provided by the Python part of Interface and used to build the 1RDM
\param result the vector of the sqrt of the occupations (n) and matrix of the NOs (no) 
              (the 1RDM = no * Diagonal(n) no.T)
              also prints the corresponding ground state energy.
*/
tuple<VectorXd, MatrixXd> Optimize_1RDM(string func, string hess_approx, string file, string cond, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
    MatrixXd overlap, MatrixXd elec1int, MatrixXd elec2int, int disp, double epsi, int Maxiter) {
    int l = overlap.rows(); int l2 = pow(l, 2);
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(l2, l2); elec2int_x = MatrixCast(T.shuffle(index), l2, l2);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E; 
    auto functional = Funcs.find(func);
    if (functional == Funcs.end()) {
        gamma.opti(&HF_func, disp, epsi, Maxiter, hess_approx, file, cond);
        E = HF_func.E(&gamma);  
    }
    else{
        gamma.opti(&functional->second, disp, epsi, Maxiter, hess_approx, file, cond);
        E = functional->second.E(&gamma);  
    }
    cout<<"E="<<E<<endl;
    return make_tuple(gamma.n(), gamma.no()); 
     
}

/*
\param arg    func: functional to use (modify this function to add new functinals)
              other arguments are provided by the Python part of Interface and used to build the 1RDM
\param result the energy of the corresponding 1RDm for the funtionla func.
*/
double E (string func, VectorXd occ, MatrixXd orbital_mat, int ne, double Enuc,
                                    MatrixXd overlap,MatrixXd elec1int, MatrixXd elec2int){
    int l = overlap.rows(); int l2 = pow(l, 2);
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(elec2int, l, l, l, l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd elec2int_x(l2, l2); elec2int_x = MatrixCast(T.shuffle(index), l2, l2);
    RDM1 gamma = RDM1(occ, orbital_mat, ne, Enuc, overlap, elec1int, elec2int, elec2int_x); double E;
    auto functional = Funcs.find(func);
    if (functional == Funcs.end()) {
        E = HF_func.E(&gamma);  
    }
    else{
        E = functional->second.E(&gamma);  
    }
    return E; 
}

/* //        USED TO TEST C++ PART 

int main(){
    int l =2; int ll = pow(l,2); double epsi =1e-6;
    VectorXd n (l); MatrixXd no (l,l); MatrixXd ovlp (l,l); MatrixXd I1 (l,l); MatrixXd I2 (ll,ll); double E_nuc; E_nuc = 0.17639240364;
    // H2 in STO-3G basis, CISD guess
    n << 0.96182731, 1.03676817;
    no<< 0.71365341, -0.70073708, -0.71365341, -0.70073708;
    ovlp<<1., 0.01826264, 0.01826264, 1. ;
    I1<<-0.64297414, -0.02083273,-0.02083273, -0.64297414;
    I2 << 7.74605944e-01, 6.54393797e-03,
        6.54393797e-03, 1.76382446e-01,
        6.54393797e-03, 1.53991591e-04,
        1.53991591e-04, 6.54393797e-03,
        6.54393797e-03, 1.53991591e-04,
        1.53991591e-04, 6.54393797e-03,
        1.76382446e-01, 6.54393797e-03,
        6.54393797e-03, 7.74605944e-01;
    Tensor<double, 4> T(l, l, l, l); T = TensorCast(I2,l,l,l,l);  Eigen::array<int, 4> index({ 1,3,0,2 });
    MatrixXd I2_x(ll, ll); I2_x = MatrixCast(T.shuffle(index), ll, ll);
    cout<<"start"<<endl;
    RDM1 gamma = RDM1(n,no,2,E_nuc,ovlp,I1,I2,I2_x);

    cout.precision(10);
    

    return 0;
}*/