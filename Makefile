CC = g++
#-- -- -- - Define names of all the object files in this project
INTELOBJS = intel - mkl.o BIN = bin LIB = lib OBJ = obj
GTEST_DIR = googletest/googletest
OBJ_DIR = obj

#Flags passed to the preprocessor.
CPPFLAGS += -isystem $(GTEST_DIR)/include

SRC =./
#-- -- -- - Define the name of the resulting released product

#-- -- -- - Define options passed by make to the compiler
CFLAGS = -g -std=c++17 
FFLAGS= -std=f2003
#-- -- -- - Setup tags for source instruction for mkl library
MKLSRCTAGS =/opt/intel/compilers_and_libraries_2018.2.199/linux/mkl/bin/mklvars.sh ia32 
SPARSKIT = $(LIB)/SPARSKIT2 
LFFLAGS = -L$(SPARSKIT)/ -L$(LIB)/

#-- -- -- - Define "all" for building the executable(s)
#-- -- -- - Include the rules for rebuilding each *.o file

#Builds gtest.a and gtest_main.a.
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h $(GTEST_HEADERS)






#
#
filecheck: 
	if [ ! -d "./bin" ]; then mkdir bin; fi
	if [ ! -d "./obj" ]; then mkdir obj; fi

$(OBJ_DIR)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -o $@ -c \
		$(GTEST_DIR)/src/gtest-all.cc

$(OBJ_DIR)/gtest_main.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -o $@ -c \
            $(GTEST_DIR)/src/gtest_main.cc

$(OBJ_DIR)/gtest.a : $(OBJ_DIR)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(OBJ_DIR)/gtest_main.a : $(OBJ_DIR)/gtest-all.o $(OBJ_DIR)/gtest_main.o
	$(AR) $(ARFLAGS) $@ $^



# All Google Test headers.  Usually you shouldn't change this definition.
GTEST_HEADERS = $(GTEST_DIR)/include/gtest/*.h \
                $(GTEST_DIR)/include/gtest/internal/*.h

taco_plugin_test:$(SRC)/taco_plugin_unit_tests.cpp
	clang++ $(CFLAGS)  $(SRC)/taco_plugin_unit_tests.cpp\
	       	-fplugin=${LLVM_ROOT}/build/lib/TacoTokensSyntax.so -o $@

unit-test: filecheck 	$(SRC)/taco_plugin_gtest.cpp $(OBJ_DIR)/gtest_main.a
	clang++ $(CFLAGS) -lpthread  -I$(GTEST_DIR)/include $(OBJ_DIR)/gtest_main.a\
	       	$(SRC)/taco_plugin_gtest.cpp \
		-fplugin=${LLVM_ROOT}/build/lib/TacoTokensSyntax.so -o bin/$@
	./bin/$@
unit-test-auto: filecheck $(SRC)/taco_plugin_gtest_auto.cpp $(OBJ_DIR)/gtest_main.a
	clang++ $(CFLAGS) -lpthread  -I$(GTEST_DIR)/include $(OBJ_DIR)/gtest_main.a\
	       	$(SRC)/taco_plugin_gtest_auto.cpp \
		-fplugin=${LLVM_ROOT}/build/lib/TacoTokensSyntax.so -o bin/$@
	./bin/$@


clean:
	rm -rf taco_plugin_test conversion_routine $(OBJ_DIR)/* bin/*

conversion_routine:$(SRC)/taco_conversion_routines.c 
	gcc $(CFLAGS) $(SRC)/taco_conversion_routines.c -L${MKLROOT}/lib/intel64 \
		-I${MKLROOT}/include -Wl,--no-as-needed -lmkl_intel_lp64\
	       	-lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl -o $@
