if(${ENABLE_GlobalArrays})
   enable_language(Fortran)
   set(TensorImpl_SRC GATensorImpl.cpp dummy.F PARENT_SCOPE)
endif()
