# TacoPlug-standalone-tests

This repository contains stand alone tests for TacoPlug: https://github.com/tpops/llvm-project-csp


# Dependency



## Build Taco Dependency
```sh
git clone https://github.com/tensor-compiler/taco.git
git checkout fb4e6dea9ac076cdb7f59697376eacf07433fdfe
cd taco
mkdir build
cd build 
cmake ../ 
make install
```
Make sure taco is successfully installed in path.

## Build Clang Compiler

```sh
git clone  https://github.com/tpops/llvm-project-csp
mkdir build
cd build
cmake -G "Unix Makefiles" -DLLVM_BUILD_EXAMPLES=OFF -DLLVM_ENABLE_PLUGINS=ON  -DCLANG_BUILD_EXAMPLES=ON -DBUILD_SHARED_LIBS=ON -DLLVM_ENABLE_PROJECTS="clang;libcxx;libcxxabi;libunwind;" ../llvm
make install
```

## Build and run test
```
 git submodule init
 git submodule update
 make unit-test
```



