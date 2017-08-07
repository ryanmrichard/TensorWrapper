foreach(name TestTensorPtr TestOperation TestShape TestMemory TestTensorWrapper
             TestTraits TestEigen TestEigenTensor TestIndices TestGAWrapper
)
    NEW_TEST(${name} UnitTests)
endforeach()
