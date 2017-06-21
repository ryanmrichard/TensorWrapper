#pragma once
#if defined(__ICC) || defined(__INTEL_COMPILER)
    _Pragma("warning(pop)")
#elif defined(__GNUC__) || defined(__GNUG__)
    _Pragma("GCC diagnostic pop")
#endif
