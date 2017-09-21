#Macro for pretty printing of options
function(option_w_default name value)
    if(${name})
        message(STATUS "Value of ${name} was set by user to : ${${name}}")
    else()
        set(${name} ${value} PARENT_SCOPE)
        message(STATUS "Setting value of ${name} to default : ${value}")
    endif()
endfunction()

#Macro for looking for dependencies/features and building them if they are not
#found (assumes the build is inside Build<NAME>.cmake in the same directory)
#one way or another a target (pkg_name)_external will be created
function(check_then_build pkg_name)
    find_package(${pkg_name} QUIET)
    if(${${pkg_name}_FOUND})
        string(TOUPPER ${pkg_name} PKG_NAME)
        message(STATUS "Found existing ${pkg_name} with")
        message(STATUS "   libraries: ${${PKG_NAME}_LIBRARIES}")
        message(STATUS "   definitions: ${${PKG_NAME}_DEFINITIONS}")
        message(STATUS "   and includes: ${${PKG_NAME}_INCLUDE_DIRS}")
        add_library(${pkg_name}_external INTERFACE)
    else()
        message(STATUS "${pkg_name} not found. Will build it for you.")
        include("Build${pkg_name}.cmake")
    endif()
endfunction()

#Utility function for stripping out variables that set to nothing (note CMake
#distinguishes between variables set to nothing and variables that are not set)
#e.g. if MPI_C_COMPILER is set to nothing the build will literally try to
#use the compiler with path ""; however, if it is not set then it will try to
#find one (assuming there's a find_package(mpi) somewhere)
function(add_extra_arg extra_arg arg_name)
    if(${arg_name})
        set(${extra_arg} ${${extra_arg}} -D${arg_name}=${${arg_name}})
    endif()
endfunction()

# Adds all source files in subdirectory "dir_name" to a variable "dir_name_SRC"
# with their full paths
#
# Note: This function assumes that in "dir_name" there is a CMake command like:
# set(dir_name_SRC source_file1.cpp source_file2.cpp... PARENT_SCOPE)
#
function(my_add_subdirectory dir_name)
    add_subdirectory(${dir_name})
    foreach(f ${${dir_name}_SRC})
        list(APPEND ${dir_name}_SRC_TMP
                    "${CMAKE_CURRENT_SOURCE_DIR}/${dir_name}/${f}")
     endforeach()
     set(${dir_name}_SRC ${${dir_name}_SRC_TMP} PARENT_SCOPE)
 endfunction()

 #Function for c-ify a variable for inclusion in the Config.hpp file
 function(c_ify in_var out_var)
     if(${${in_var}})
         set(${out_var} 1 PARENT_SCOPE)
     else()
         set(${out_var} 0 PARENT_SCOPE)
     endif()
 endfunction()
