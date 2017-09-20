#pragma once
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__ICC) || defined(__INTEL_COMPILER)
    _Pragma("warning(push)")
    //_Pragma("warning(disable:869)")

#elif defined(__GNUC__) || defined(__GNUG__)
    _Pragma("GCC diagnostic push")
    _Pragma("GCC diagnostic ignored \"-Wenum-compare\"")
    _Pragma("GCC diagnostic ignored \"-Wnarrowing\"")
    _Pragma("GCC diagnostic ignored \"-Wsign-compare\"")
#endif


#ifdef __cplusplus
}
#endif
