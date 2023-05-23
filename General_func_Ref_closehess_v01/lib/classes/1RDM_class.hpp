#ifndef _1RDM_CLASS_hpp_
#define _1RDM_CLASS_hpp_
#include "Functional_class.hpp"
#include<tuple>
#include <vector>
using namespace std;

class Functional;
;

// Defines the 1RDM class (see 1RDM_class.cpp for more detail)
class RDM1{
    private:
        VectorXd V_;                           //Sum of derivatives of erf (used to compute derivatives ; vector if subspaces)
        VectorXd W_;                           //Sum of 2nd derivaties of erf (used to compute 2nd derivative)
        VectorXi computed_V_;                  //Wether V is up to date  (vector if subspaces)
        VectorXi computed_W_;;                 //Wether W is up to date  (vector if subspaces)
        VectorXd x_;                           //Vector parametrising the occupations 
        MatrixXd deviation_;                   //Variation of the NO matrix : used to numerically compute the NO part of the auxiliary gradient 
        MatrixXd no_;                          //Initial matrix of natural orbitals
    public:
        
        int n_elec;                            //Number of electrons
        double E_nuc;                          //Nuclear energy
        MatrixXd ovlp;                         //Overlap matrix
        VectorXd mu;                           //EBI variable (vector if subspaces)
        VectorXd l_theta;                      //List of parameters of U / NO=U.NO
        vector<vector<int>> omega;             //Indicies of the subspace repartition
        void set_n (VectorXd);
        void set_n(int,double); 
        void set_deviation(int i, int j, double epsi){deviation_(i,j) += epsi;};
        MatrixXd deviation(){return deviation_;};
        MatrixXd  no0() const {return no_;};
        MatrixXd   no() const;
        MatrixXd  dno(int) const;
        double get_V(int g=0);
        double get_W(int g=0);
        void solve_mu();
        double n(int) const;
        VectorXd n() const;
        double x(int i) const { return x_(i);};// Return x_i
        VectorXd x() const { return x_;};      // Return x
        void x (int, double);
        void x (VectorXd);
        int size() const { return x_.size(); };//Return the number of NOs
        double dn(int,int); VectorXd dn(int);
        double ddn(int,int,int); VectorXd ddn(int,int);
        double dsqrt_n(int,int); VectorXd dsqrt_n(int);
        double ddsqrt_n(int,int,int); VectorXd ddsqrt_n(int,int);
        MatrixXd int1e; MatrixXd int2e; MatrixXd int2e_x;
        int find_subspace(int) const;
        RDM1();
        RDM1(int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(VectorXd,MatrixXd,int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(const RDM1*);
        ~RDM1();
        MatrixXd mat() const;
        MatrixXd matL() const;
        MatrixXd matR() const;
        MatrixXd dmat(int);
        void opti(Functional*,int disp=0,double epsi=1e-6,int maxiter=100, string hess_approx="exa", string file = "test", string cond="xtol");
};
//Auxiliary function used to minimise the energy of the 1RDM
tuple<double,int> opti_aux(RDM1*,Functional*, string, ofstream*, ofstream*, string, double epsilon=1e-8,bool disp=false, int maxiter=100);

//Structure used to pass the parameters (functional, 1RDM, auxiliary hessian,...) to the fides optimiser
typedef struct{
    RDM1* gamma; Functional* func; int g; int niter; VectorXd x1; VectorXd x2; VectorXd grad1; VectorXd grad2; MatrixXd hess_; MatrixXd Hanaly; string hess_approx; 
    ofstream* ofile; ofstream* ofile_behav; bool exa_nn; bool exa_NONO; bool exa_nNO; bool do_1st_iter; bool disp; bool update_hess; int interval;
    }data_struct;

//Hessian updates
void BFGS (void*)   ; void SR1 (void*); void H_init (MatrixXd*,VectorXd,VectorXd);
void BFGS_aux(void*); void SR1_aux(void*); void DFP_aux(void*); 

//Other auxiliary functions
MatrixXd Givens(MatrixXd,const VectorXd*); MatrixXd dGivens(MatrixXd, const VectorXd*,int); MatrixXd ddGivens(MatrixXd, const VectorXd*,int,int);
void print_t(chrono::high_resolution_clock::time_point, chrono::high_resolution_clock::time_point, int iter=1);
VectorXd negative_eigvls(MatrixXd M,double epsi=1e-8);
#endif