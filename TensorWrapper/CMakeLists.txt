cmake_minimum_required(VERSION 3.2)
project(TensorWrapper-Core CXX)
include(${TWRAPPER_ROOT}/cmake/cmake_macros.cmake)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS False)

# CMake doesn't support Intel CXX standard until cmake 3.6
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
  if("${CMAKE_VERSION}" VERSION_LESS "3.6")
      add_compile_options(-std=c++${CMAKE_CXX_STANDARD})
  endif()
endif()

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
foreach(pkg ${EXTERNAL_FEATURES})
    find_package(${pkg} REQUIRED)
    set(ENABLE_${pkg} TRUE)
    list(APPEND EXTERNAL_INCLUDE_DIRS ${${pkg}_INCLUDE_DIRS})
    list(APPEND EXTERNAL_LIBRARIES ${${pkg}_LIBRARIES})
endforeach()
###############################
#  Do we need MPI?
##############################
if(${ENABLE_GlobalArrays})
    find_package(MPI REQUIRED)
    message(STATUS ${MPI_CXX_INCLUDE_PATH})
    list(APPEND EXTERNAL_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
    list(APPEND EXTERNAL_LIBRARIES ${MPI_CXX_LIBRARIES})
endif()


include_directories(${EXTERNAL_INCLUDE_DIRS})
my_add_subdirectory(TensorImpl)

# Whether we have source to compile or not depends on what features are enabled
if(TensorImpl_SRC)
    #This is kindof a hack...
    enable_language(Fortran)
    message(STATUS "Building an actual library.${TensorImpl_SRC}.")
    list(APPEND FORTRAN_LIBS ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
    foreach(lib ${CMAKE_Fortran_IMPLICIT_LINK_LIBRARIES})
        find_library (${lib}_PATH lib${lib}.so ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES})
        list(APPEND Fortran_EXTERNAL_LIBRARIES ${${lib}_PATH})
    endforeach()
    add_library(TensorWrapper ${TensorImpl_SRC})
    target_link_libraries(TensorWrapper PUBLIC ${Fortran_EXTERNAL_LIBRARIES})
    target_compile_definitions(TensorWrapper PUBLIC ${EXTERNAL_DEFINES})
else()
    add_library(TensorWrapper INTERFACE)
    target_link_libraries(TensorWrapper INTERFACE ${EXTERNAL_LIBRARIES})
    target_include_directories(TensorWrapper INTERFACE ${TWRAPPER_ROOT}
                                                    ${EXTERNAL_INCLUDE_DIRS}
    )
endif()



install(TARGETS TensorWrapper
        LIBRARY DESTINATION lib/
        ARCHIVE DESTINATION lib/
        RUNTIME DESTINATION lib/)

# Install all headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        DESTINATION include
        FILES_MATCHING
        REGEX ".*\.h(pp)?$")

# Write the configure file so other packages can find us
configure_file("${TWRAPPER_ROOT}/cmake/TensorWrapperConfig.cmake.in"
               TensorWrapperConfig.cmake @ONLY)

# Install said file
install(FILES ${CMAKE_BINARY_DIR}/TensorWrapperConfig.cmake
        DESTINATION share/cmake/TensorWrapper)
