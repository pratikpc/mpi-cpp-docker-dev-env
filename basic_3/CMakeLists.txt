cmake_minimum_required(VERSION 3.12)

project(basic_3)

add_executable(${PROJECT_NAME} main.cpp)

# Require MPI for this project:
find_package(MPI REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
