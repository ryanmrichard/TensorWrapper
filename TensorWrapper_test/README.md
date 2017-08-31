TensorWrapper_test
==================

This directory contains TensorWrapper's testing machinery.

In `TestHelpers.cpp` you will find the definitions of various functions used
throughout the tests.

In `StressTests` you will find tests geared at:
 - ensuring that the TensorWrapper API has minimal overhead and
 - assessing the scalability of the backends.

In `UnitTests` you will find tests that are designed to ensure that each class's
member functions are operating correctly.
