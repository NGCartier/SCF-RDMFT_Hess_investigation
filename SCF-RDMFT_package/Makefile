PYEXT = $(shell python3-config --extension-suffix)
PYFILE = -I$(shell whereis python3)
EIGFILE = -I/usr/local/Cellar/eigen/3.4.0_1/include/eigen3
FLAGS = -Wno-unqualified-std-cast-call -shared -Ofast -fsigned-zeros -std=c++17 -undefined dynamic_lookup $(python3 -m pybind11 --includes) $(PYFILE) $(EIGFILE) -opt/local/include/lapack/
LAPACK = -framework Accelerate /opt/local/lib/lapack/liblapacke.dylib -D EIGEN_USE_LAPACKE=1 -lblas -llapack # LAPACK is faster but less stable.
FILES = ./lib/Interface.cpp ./lib/classes/1RDM_class.cpp ./lib/classes/Hessian.cpp ./lib/classes/Functional_class.cpp ./lib/numerical_deriv/numerical_deriv.cpp ./lib/classes/EBI_add.cpp ./lib/tools.cpp
FUNCS:= $(shell ls ./lib/Functionals/*.cpp)
FLAGS2 = -lfmt -lm
Compute_1RDM.$(PYEXT): 
	c++  $(FLAGS) $(FILES) $(FUNCS) $(FLAGS2) -o Compute_1RDM$(PYEXT)
clean:
	rm Compute_1RDM.$(PYEXT)
