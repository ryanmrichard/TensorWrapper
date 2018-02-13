cmake_minimum_required(VERSION ${CMAKE_VERSION})
project(TensorWrapper-SRC VERSION ${PROJECT_VERSION} LANGUAGES CXX)
include(TargetMacros)
set(TensorWrapper_SRCS Space.cpp)
set(TensorWrapper_INCLUDES Space.hpp)
set(TensorWrapper_DEFINITIONS )
set(TensorWrapper_LINK_FLAGS )
nwchemex_add_library(TensorWrapper TensorWrapper_SRCS
                                   TensorWrapper_INCLUDES
                                   TensorWrapper_DEFINITIONS
                                   TensorWrapepr_LINK_FLAGS)
