#include "GATensor.hpp"
#include <macdecls.h>
#include <mpi.h>

void* replace_malloc(size_t bytes, int align, char *name)
{
    return malloc(bytes);
}

void replace_free(void *ptr)
{
    free(ptr);
}

namespace TWrapper {
namespace detail_ {

//Exists basically to move initialization/finalization to RAII model
struct GAEnv {
    int was_mpi_already_started_;
    GAEnv(int heap, int stack)
    {
       //TODO: Take an MPI communicator
        MPI_Initialized(&was_mpi_already_started_);
        if(!was_mpi_already_started_)
            MPI_Init(nullptr,nullptr);
        NGA_Initialize();
        MA_init(C_DBL,stack,heap);
        GA_Register_stack_memory(replace_malloc, replace_free);
    }

    ~GAEnv()
    {
        GA_Terminate();
        if(!was_mpi_already_started_)
            MPI_Finalize();
    }

};

std::unique_ptr<GAEnv> Env_;

}//End namespace detail_

void GAInitialize(int heap,int stack){
    if(!detail_::Env_)
        detail_::Env_=std::make_unique<detail_::GAEnv>(heap,stack);
}

}//End namespace TWRapper
