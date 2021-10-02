cmake_minimum_required(VERSION 3.12)

project(1diff_par)
find_package(MPI REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
