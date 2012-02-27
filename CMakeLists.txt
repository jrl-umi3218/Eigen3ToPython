cmake_minimum_required(VERSION 2.8)
project(Eigen3ToPython)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -std=c++0x -pedantic")

find_package(PkgConfig REQUIRED)

add_subdirectory(src)

