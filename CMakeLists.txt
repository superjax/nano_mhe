cmake_minimum_required(VERSION 2.8.3)
project(nano_mhe)

set(CMAKE_CXX_STANDARD 17)

enable_testing()
find_package(Eigen3 REQUIRED)

find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)

include_directories(include
    lib/geometry/include
    lib/utils/include
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS})

add_library(utils
    lib/utils/src/robot1d.cpp)

add_executable(nano_mhe_test
    src/test_imu1d.cpp
    src/test_mhe1d.cpp
    src/test_autodiff.cpp
    src/test_lm.cpp)
target_link_libraries(nano_mhe_test
    utils
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    pthread)
#add_custom_command(
#     TARGET nano_mhe_test
#     COMMENT "Run tests"
#     POST_BUILD
#     COMMAND nano_mhe_test)
