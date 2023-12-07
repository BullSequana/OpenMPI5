/*
 * Copyright (c) 2012      Oak Rigde National Laboratory. All rights reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/memchecker.h"
#include "ompi/mpiext/pcoll_part/c/mpiext_pcoll_part_c.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Pbcast_init = PMPIX_Pbcast_init
#endif
#define MPIX_Pbcast_init PMPIX_Pbcast_init
#endif

static const char FUNC_NAME[] = "MPIX_Pbcast_init";


int MPIX_Pbcast_init(void *buffer, size_t parts, size_t count, MPI_Datatype datatype,
                    int root, MPI_Comm comm, MPI_Info info, MPI_Request *request)
{
    int err;
    int total_count = parts * count;

    //SPC_RECORD(OMPI_SPC_BCAST_INIT, 1);

    MEMCHECKER(
        // cppcheck-suppress unknownMacro
        memchecker_datatype(datatype);
        memchecker_call(&opal_memchecker_base_isdefined, buffer, total_count, datatype);
        memchecker_comm(comm);
    );

    if (MPI_PARAM_CHECK) {
      err = MPI_SUCCESS;
      OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
      if (ompi_comm_invalid(comm)) {
          return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM,
                                     FUNC_NAME);
      }

      /* Errors for all ranks */
      OMPI_CHECK_DATATYPE_FOR_SEND(err, datatype, total_count);
      OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
      if (MPI_IN_PLACE == buffer) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
      }

      /* Errors for intracommunicators */
      if (OMPI_COMM_IS_INTRA(comm)) {
        if ((root >= ompi_comm_size(comm)) || (root < 0)) {
          return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
        }
      }
      /* Errors for intercommunicators */
      else {
        if (! ((root >= 0 && root < ompi_comm_remote_size(comm)) ||
               MPI_ROOT == root || MPI_PROC_NULL == root)) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
        }
      }
    }

    /* Invoke the coll component to perform the back-end operation */
    if (OPAL_UNLIKELY(NULL == comm->c_coll->coll_pbcast_init)) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_UNSUPPORTED_OPERATION, FUNC_NAME);
    }

    err = comm->c_coll->coll_pbcast_init(buffer, parts, count, datatype, root, comm,
                                        info, request,
                                        comm->c_coll->coll_pbcast_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        if (!OMPI_COMM_IS_INTRA(comm)) {
            if (MPI_PROC_NULL == root) {
                datatype = NULL;
            }
        }
        ompi_coll_base_retain_datatypes(*request, datatype, NULL);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
