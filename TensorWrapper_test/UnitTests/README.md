UnitTests
=========

This directory contains tests aimed at ensuring that the classes and functions
within TensorWrapper are working as expected.  These tests are designed to run
fast and to isolate functionality (*i.e.* the interfaces used throughout the
tests do not reflect the public APIs, but rather are "bare metal" invocations).
Below is a list of tests and what they test

- TestEigen ensures that the Eigen matrix/vector backend is wrapped correctly
- TestEigenTensor ensures that Eigen's tensor class is wrapped correctly
- TestGAWrapper ensures that the Global Arrays backend is wrapped correctly
- TestIndices ensures compile time index parsing is working correctly
- TestMemory tests related to the MemoryBlock class are here
- TestOperation ensures lazy evaluation works
- TestShape tests the Shape class
- TestTensorPtr focuses on tests of the type-erasing TensorPtr class
- TestTensorWrapper tests our public API
- TestTiledArray tests that the Tiled Array backend is wrapped correctly
- TestTraits ensures our meta-template programming is right
