cmake_minimum_required(VERSION 3.2)
project(TensorWrapper-test CXX)
find_package(TensorWrapper REQUIRED)
include(CTest)
enable_testing()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS False)

# CMake doesn't support Intel CXX standard until cmake 3.6
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  if("${CMAKE_VERSION}" VERSION_LESS "3.6")
      add_compile_options(-std=c++${CMAKE_CXX_STANDARD})
  endif()
endif()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(TEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

function(NEW_TEST test_name test_dir)
   add_executable(${test_name} ${test_name}.cpp)
   target_link_libraries(${test_name} PRIVATE TensorWrapper)
   target_include_directories(${test_name} PRIVATE ${TEST_ROOT})
   add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
   install(TARGETS ${test_name} DESTINATION ${test_dir})
endfunction()

foreach(dir StressTests UnitTests)
    add_subdirectory(${dir})
    install(FILES ${CMAKE_BINARY_DIR}/${dir}/CTestTestfile.cmake DESTINATION ${dir})
endforeach()

# This file was created from the various add_test commands
install(FILES ${CMAKE_BINARY_DIR}/CTestTestfile.cmake DESTINATION .)
