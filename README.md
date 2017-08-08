TensorWrapper
=============

[![Documentation](https://codedocs.xyz/ryanmrichard/TensorWrapper.svg)](https://codedocs.xyz/ryanmrichard/TensorWrapper/)

Brief Description
-----------------

The TensorWrapper project aims to provide a unified, unchanging API to many
of the existing C and C++ Tensor projects (see [supported backends]
for a list of currently supported tensor libraies).  This aim is primarily
motivated in an effort to facilitate interoperability between libraries that use
different tensor backends.  However, this library also provides a mechanism for:

- Decoupling new codes from the tensor backend
- Faciliating the porting of codes to multiple platforms (well the tensor part
  anyways)
- Rapidly trying out many different tensor libraries to figure out which works
  best for your project.


Additional Resources
--------------------

This README is really just the portal to the TensorWrapper documentation.  Below
are a variety of resources for using and extending the TensorWrapper library.

[Configuring and Compiling](/dox/Building.md)

[QuickStart](dox/QuickStart.md)

[Supported Tensor Library Backends](/dox/SupportedBackends.md)

<!-- Links -->
[supported backends]: /dox/SupportedBackends.md
