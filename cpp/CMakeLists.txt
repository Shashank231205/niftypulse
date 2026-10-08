cmake_minimum_required(VERSION 3.20)
project(niftypulse_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(MSVC)
  add_compile_options(/W4 /permissive- /O2)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

add_library(niftypulse_core INTERFACE)
target_include_directories(niftypulse_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(replay_bench src/replay_bench.cpp)
target_link_libraries(replay_bench PRIVATE niftypulse_core)

option(NIFTYPULSE_BUILD_PYTHON "Build the pybind11 module" OFF)
if(NIFTYPULSE_BUILD_PYTHON)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(niftypulse_native src/bindings.cpp)
  target_link_libraries(niftypulse_native PRIVATE niftypulse_core)
endif()
