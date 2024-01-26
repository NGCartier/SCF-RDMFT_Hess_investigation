#ifndef _1RDM_CLASS_hpp_
#define _1RDM_CLASS_hpp_
#include "Functional_class.hpp"
#include <tuple>
#include <vector>
#include <deque>
using namespace std;

class Functional; class optimizer;

//Structure used to pass the functional and 1RDM to optimizer methods
typedef struct{
    RDM1* gamma; Functional* func; int g; MatrixXd NO; int niter; ofstream* ofile; bool disp;
    string hess_approx; MatrixXd hess_; MatrixXd hess_exp_; MatrixXd hess_cheap_; 
    VectorXd x1; VectorXd x2; VectorXd grad1; VectorXd grad2; double E1; double E2;
    deque<VectorXd> lstep; deque<VectorXd> ly; int memory; double epsi;
    bool exa_nn; bool exa_NONO; bool exa_nNO; bool do_1st_iter; bool update_hess; bool mixed; int interval;
}data_struct;
 
// Defines the 1RDM class (see 1RDM_class.cpp for more detail)
class RDM1{
    private:
        VectorXd V_;                           //Sum of derivatives of erf (used to compute derivatives ; vector if subspaces)
        VectorXd W_;                           //Sum of 2nd derivaties of erf (used to compute 2nd derivative) 
        VectorXd x_;                           //Vector parametrising the occupations
        VectorXi computed_V_;                  //Wether V is up to date  (vector if subspaces) 
        VectorXi computed_W_;                  //Wether W is up to date  (vector if subspaces)
        
    public:
        int n_elec;                            //Number of electrons
        MatrixXd no;                           //Matrix of natural orbitals
        double E_nuc;                          //Nuclear energy
        MatrixXd ovlp;                         //Overlap matrix
        VectorXd mu;                           //EBI variable (vector if subspaces)
        vector<vector<int>> omega;             //Indicies of the subspace repartition
        void set_n (VectorXd);
        void set_n(int,double); 
        void set_sqrtn(VectorXd);
        void set_sqrtn(int,double);
        void set_no(MatrixXd no0){no=no0; };  // Affect no0 to no
        double get_V(int g=0);
        double get_W(int g=0);
        void solve_mu();
        void subspace();
        double n(int) const;
        VectorXd n() const;
        double sqrtn(int) const;
        VectorXd sqrtn () const;
        double x(int i) const { return x_(i);};// Return x_i
        VectorXd x() const { return x_;};      // Return x
        void x (int, double);
        void x (VectorXd);
        int size() const { return x_.size(); };//Return the number of NO
        double dn(int,int); MatrixXd dn(int);
        double ddn(int,int,int); MatrixXd ddn(int,int);
        double dsqrt_n(int,int); MatrixXd dsqrt_n(int);
        double ddsqrt_n(int,int,int); MatrixXd ddsqrt_n(int,int);
        MatrixXd int1e; MatrixXd int2e; MatrixXd int2e_x;
        int find_subspace(int) const;
        RDM1();
        RDM1(int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(VectorXd,MatrixXd,int,double,MatrixXd,MatrixXd,MatrixXd,MatrixXd);
        RDM1(const RDM1*);
        ~RDM1();
        MatrixXd mat() const;
        void opti(Functional*, string hess_approx="exa", string file="", int disp=0,double epsi=1e-6, double epsi_n=1e-3, int maxiter=100);
};
//Auxiliary functions used to minimise the energy of the 1RDM
double norm2(VectorXd* x); double norm1(VectorXd* x);
tuple<double,int> opti_nno( RDM1*,Functional*,string,ofstream*,double epsilon=1e-8, bool disp=false, int maxiter=100);
MatrixXd exp_unit(VectorXd*); 
//Other auxiliary functions
VectorXd negative_eigvls(MatrixXd M,double epsi=1e-8);
tuple<VectorXd,MatrixXd,VectorXd,MatrixXd> negative_eigvects(MatrixXd M,double epsi=1e-8);

//Hessian updates
void SR1 (void*); void BFGS (void*); void DFP (void*); void Broyden (void*); void LBFGS (void*); void ZERO (void*); 
void SR1_aux (void*); void BFGS_aux(void*); void tBFGS_aux(void*); void sBFGS_aux(void*); void LBFGS_aux(void*); void LbBFGS_aux(void*);
void H_init (MatrixXd*,VectorXd,VectorXd,int len=0); 


//Numerical constants
const double  SQRT_TWO = 1.4142135623730950;
const double RSQRT_TWO = 0.7071067811865475;
const double   OCC_ONE = 0.3853393505171266; // s.t. x_p = OCC_ONE - mu for n_p = 1
    //to parametrise the activation function
const double BETA  = 1.; const double EPSILON = 1e-3; const double ZETA = 0.1;

#endif
