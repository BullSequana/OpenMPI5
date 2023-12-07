/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012      Oak Ridge National Laboratory. All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2014-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
#pragma weak MPIX_Palltoall_init = PMPIX_Palltoall_init
#endif
#define MPIX_Palltoall_init PMPIX_Palltoall_init
#endif

static const char FUNC_NAME[] = "MPIX_Palltoall_init";


int MPIX_Palltoall_init(const void *sendbuf, int sendparts, int sendcount, MPI_Datatype sendtype,
                       void *recvbuf, int recvparts, int recvcount, MPI_Datatype recvtype,
                       MPI_Comm comm, MPI_Info info, MPI_Request *request)
{
    size_t sendtype_size, recvtype_size;
    int err;
    int total_scount = sendparts * sendcount;
    int total_rcount = recvparts * recvcount;

    SPC_RECORD(OMPI_SPC_PALLTOALL_INIT, 1);

    MEMCHECKER(
        // cppcheck-suppress unknownMacro
        memchecker_comm(comm);
        if (MPI_IN_PLACE != sendbuf) {
            memchecker_datatype(sendtype);
            memchecker_call(&opal_memchecker_base_isdefined, (void *)sendbuf, total_scount, sendtype);
        }
        memchecker_datatype(recvtype);
        memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, total_rcount, recvtype);
    );

    if (MPI_PARAM_CHECK) {

        /* Unrooted operation -- same checks for all ranks on both
           intracommunicators and intercommunicators */

        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM,
                                          FUNC_NAME);
        } else if ((MPI_IN_PLACE == sendbuf && OMPI_COMM_IS_INTER(comm)) ||
                   MPI_IN_PLACE == recvbuf) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_ARG,
                                          FUNC_NAME);
        } else {
            if (MPI_IN_PLACE != sendbuf) {
                OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, total_scount);
                OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
            }
            OMPI_CHECK_DATATYPE_FOR_RECV(err, recvtype, total_rcount);
            OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
        }

        if (MPI_IN_PLACE != sendbuf && !OMPI_COMM_IS_INTER(comm)) {
            ompi_datatype_type_size(sendtype, &sendtype_size);
            ompi_datatype_type_size(recvtype, &recvtype_size);
            if ((sendtype_size*total_scount) != (recvtype_size*total_rcount)) {
                fprintf(stderr, "Send %dx%d Recv %dx%d\n",sendtype_size, total_scount,
    recvtype_size, total_rcount); fflush(stderr);
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TRUNCATE, FUNC_NAME);
            }
        }
    }

    if (OPAL_UNLIKELY(NULL == comm->c_coll->coll_palltoall_init)) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_UNSUPPORTED_OPERATION, FUNC_NAME);
    }

    /* Invoke the coll component to perform the back-end operation */
    err = comm->c_coll->coll_palltoall_init(sendbuf, sendparts, sendcount, sendtype,
                                           recvbuf, recvparts, recvcount, recvtype, comm, info,
                                           request, comm->c_coll->coll_palltoall_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_datatypes(*request, (MPI_IN_PLACE==sendbuf)?NULL:sendtype, recvtype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
