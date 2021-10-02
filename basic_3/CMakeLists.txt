cmake_minimum_required(VERSION 3.12)

project(basic_3)

add_executable(${PROJECT_NAME} main.cpp)

# Require MPI for this project:
find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_CXX)
