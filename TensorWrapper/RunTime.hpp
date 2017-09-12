#pragma once
#include "TensorImpls.hpp"
/** \file Contains the definition and implementation of the RunTime class.
 *
 */

namespace TWrapper {

/** \brief A class for holding the details of the run time enviornment.
 *
 *  In particular this class is concerned with the number of MPI processes.
 *  It also will call the appropriate start-up functions for each backend.
 */
class RunTime{
public:

    RunTime(int argc=0, char** argv=nullptr)
    {
    #ifdef ENABLE_tiledarray
        TiledArray::initialize(argc,argv);
    #endif
    #ifdef ENABLE_GAXX
        GAInitialize();
    #endif
    }

    ~RunTime()
    {
    #ifdef ENABLE_tiledarray
        TiledArray::finalize();
    #endif
    }

};

}//End namespace
