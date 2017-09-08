foreach(test_name BLASBaseLine StressTestEigen StressTestEigenTensor
                  StressTestGlobalArrays CCSD StressTestTiledArray)
    NEW_TEST(${test_name} StressTests)
endforeach()

