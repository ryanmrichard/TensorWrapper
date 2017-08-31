Adding a New Backend
====================

The purpose of this page is to document how to add a new backend to
TensorWrapper.

Update the Build
----------------

To this end we need to assign an unique identifier to the backend to be used
throughout the build.  In general this identifier should be whatever needs to be
passed to `find_package` (which is case sensitive) for it to be found. Whatever
that identifier is we refer to it as `pkg` throughout this section, so whenever
you see `pkg` from now on replace it with your case-sensitive identifier.  As a
general note we will occasionally need the all capital letter version of `pkg`
and we denote this by `PKG`.

Step 0.  If your backend is not compatible with cmake's `find_package` system
you will need to write a `Findpkg.cmake` file and place it in the `cmake`
directory.  This file should find the necessary includes and libraries for the
backend (assigning them to `PKG_INCLUDE_DIRS` and `PKG_LIBRARIES` respectively).
Furthermore, if there are any macro definitions needed to enable/disable
functionality for the backend they should be assigned to `PKG_DEFINITIONS`.

Step 1. With the identifier in hand, in the `external` directory create a file
`Buildpkg`.  In general this file should download and compile the source code of
the backend forwarding the necessary compile flags and options to it.  This file
may be blank which will force the user to compile and install the backend before
it can be used with TensorWrapper.

Step 2.  In the top-level `CMakeLists.txt` add a new option `ENABLE_pkg` in the
backends section that defaults to false.  Then a few lines down, add `pkg` to
the list of backends being assinged to the variable `XXX`.

Register the Backend
--------------------

The directions in this section all talke place in the `TensorWrapper/TensorImpl`
directory unless otherwise specified.

Step 1.  Each backend must have an enum associated with it in `TensorTypes.hpp`.
Add the backend to the TensorTypes enum and also create an entry for it further
down the header file so it can be found when looping over backends.

Step 2.  Partially specialize `TensorWrapperImpl` for your new enum.  In
particular you should ensure your class has the following methods (even if they
just throw because they are not implemented).  Full signatures and stipulations
are available in the documentation for the primary `TensorWrapperImpl` template.

   - `dims` returns the shape of the tensor
   - `get_memory` returns the raw local memory of a tensor.
   - `set_memory` sets the raw local memory of a tensor
   - `add` returns the sum of two tensors (or an object that can do said sum)
   - `subtract` same as `add`, but for subtraction
   - `scale` scales a tensor by a constant
   - `are_equal` returns true if the elements of two tensors are exactly equal
   - `eval` given a tensor or lazy evaluation object return the result
   - `permute` shuffles the dimension orders
   - `contraction` contracts two tensors

Step 3. If the backend requires initialization/finalization add it to the
RunTime class.
