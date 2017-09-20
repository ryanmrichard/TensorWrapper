TensorWrapper
=============

[![Join the chat at https://gitter.im/TensorWrapper/Lobby](https://badges.gitter.im/TensorWrapper/Lobby.svg)](https://gitter.im/TensorWrapper/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

[![Documentation](https://codedocs.xyz/ryanmrichard/TensorWrapper.svg)](https://codedocs.xyz/ryanmrichard/TensorWrapper/)
[![Build Status](https://travis-ci.org/ryanmrichard/TensorWrapper.svg?branch=master)](https://travis-ci.org/ryanmrichard/TensorWrapper)

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

[Linear Algebra API](dox/LinAlg.md)

<!-- Links -->
[supported backends]: /dox/SupportedBackends.md
