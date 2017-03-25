cmake_minimum_required(VERSION 3.6)
project(ML_Regression)

#set(CMAKE_COMPILER_IS_GNUCC)
#set(CMAKE_CXX_COMPILER "Users/elenasolovyeva/Applications/Xcode.app/Contents/Developer/usr/bin/gcc")

set(CMAKE_CXX_STANDARD 11)

set(EIGEN_ROOT "/Users/elenasolovyeva/Documents/Cpp/Eigen")
include_directories(${EIGEN_ROOT})


set(SOURCE_FILES main.cpp
        EquityPriceGenerator.h
        EquityPriceGenerator.cpp)
add_executable(ML_Regression ${SOURCE_FILES})