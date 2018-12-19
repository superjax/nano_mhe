cmake_minimum_required(VERSION 2.8.3)
project(nano_mhe)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")

find_package(Eigen3 REQUIRED)
find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)

include_directories(include
    lib
    ${EIGEN3_INCLUDE_DIRS})

add_library(nano_mhe
  src/nano_mhe.cpp
  include/nano_mhe.h)

add_executable(nano_mhe_test src/test.cpp)
target_link_libraries(nano_mhe_test nano_mhe gtest gtest_main ${GTEST_LIBRARIES} pthread)

