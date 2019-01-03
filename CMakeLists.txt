cmake_minimum_required(VERSION 2.8.3)
project(nano_mhe)

set(CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")

enable_testing()
find_package(Eigen3 REQUIRED)
#add_subdirectory(lib/eigen3)
#set(EIGEN3_INCLUDE_DIRS "lib/eigen3 lib/eigen3/Eigen lib/eigen3/unsupported")

find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)

include_directories(include
    lib/geometry/include
    lib/utils/include
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS})

add_library(utils
    lib/utils/include/utils/robot1d.h
    lib/utils/include/utils/logger.h
    lib/utils/src/robot1d.cpp)

add_library(nano_mhe
  src/nano_mhe.cpp
  include/nano_mhe1d.h
  include/imu1d.h
  include/pos1d.h
  include/vel1d.h
  include/nano_ad.h
  include/nano_lm.h
)

add_executable(nano_mhe_test
    src/test_imu1d.cpp
    src/test_mhe1d.cpp
    src/test_autodiff.cpp
    src/test_lm.cpp
    include/test_common.h)
target_link_libraries(nano_mhe_test utils nano_mhe gtest gtest_main ${GTEST_LIBRARIES} pthread)
#add_custom_command(
#     TARGET nano_mhe_test
#     COMMENT "Run tests"
#     POST_BUILD
#     COMMAND nano_mhe_test)
