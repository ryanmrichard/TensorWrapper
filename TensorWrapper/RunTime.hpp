#pragma once
#include "TensorImpls.hpp"
/** \file Contains the definition and implementation of the RunTime class.
 *
 */
//#ifndef ENABLE_CTF
//namespace CTF{
//    using World=int;
//}
//#endif

namespace TWrapper {
//namespace detail_ {
//    template<typename Dummy>
//    struct DaWorld{
//        static std::unique_ptr<CTF::World> world_;
//    };

//    template<typename Dummy>
//    std::unique_ptr<CTF::World> DaWorld<Dummy>::world_;
//}




/** \brief A class for holding the details of the run time enviornment.
 *
 *  In particular this class is concerned with the number of MPI processes.
 *  It also will call the appropriate start-up functions for each backend.
 */
class RunTime { //private detail_::DaWorld<void> {
    bool initialized_=false;
public:    
    //static CTF::World& world(){return *world_;}
    RunTime(int argc=0, char** argv=nullptr, MPI_Comm comm=MPI_COMM_WORLD)
    {
    #ifdef ENABLE_CTF
        int temp_init;
        MPI_Initialized(&temp_init);
        initialized_=temp_init!=0;
        if(!initialized_)
        {
            MPI_Init(&argc,&argv);
            comm=MPI_COMM_WORLD;
        }
        //world_=std::make_unique<CTF::World>(argc,argv);
    #endif
    #ifdef ENABLE_tiledarray
        TiledArray::initialize(argc,argv);
    #endif
    #ifdef ENABLE_GAXX
        GAInitialize();
    #endif
    }

    ~RunTime()
    {
        //world_.swap(std::unique_ptr<CTF::World>());
    #ifdef ENABLE_tiledarray
        TiledArray::finalize();
    #endif
    #ifdef ENABLE_CTF
        if(initialized_)
        {
            MPI_Finalize();
        }
    #endif
    }

};

}//End namespace
