if(${ENABLE_GlobalArrays})
    NEW_TEST(TestGATensor UnitTests)
    NEW_TEST(TestGAWrapper UnitTests)
endif()
foreach(name TestEigen TestEigenTensor TestContraction TestConversions
             TestShape TestMemory TestTensorPtr
             TestOperation
)
    NEW_TEST(${name} UnitTests)
endforeach()
