if(${ENABLE_STRESS_TESTS})
    foreach(test_name BLASBaseLine StressTestEigen StressTestEigenTensor
            CCSD)
        NEW_TEST(${test_name} StressTests)
    endforeach()
endif()
