#ifndef _TOOLS_HPP_
#define _TOOLS_HPP_
using namespace Eigen;

void print_t(std::chrono::high_resolution_clock::time_point, std::chrono::high_resolution_clock::time_point, int iter=1);
double sign(double);
VectorXd pow(const VectorXd*, double); VectorXd pow(VectorXd, double);

#endif