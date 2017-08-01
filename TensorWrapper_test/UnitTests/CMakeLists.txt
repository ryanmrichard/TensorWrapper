if(${ENABLE_GlobalArrays})
    NEW_TEST(TestGATensor UnitTests)
    NEW_TEST(TestGAWrapper UnitTests)
endif()
foreach(name TestTensorPtr TestOperation TestShape TestMemory TestTensorWrapper
             TestTraits TestEigen TestEigenTensor TestIndices
)
    NEW_TEST(${name} UnitTests)
endforeach()
