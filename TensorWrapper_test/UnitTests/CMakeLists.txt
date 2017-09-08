foreach(name TestTensorPtr TestOperation TestShape TestMemory TestTensorWrapper
             TestTraits TestEigen TestEigenTensor TestIndices TestGAWrapper
             TestTiledArray TestTypeComparisons
)
    NEW_TEST(${name} UnitTests)
endforeach()
