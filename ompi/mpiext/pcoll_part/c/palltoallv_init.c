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
  Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2007      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2012-2013 Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2014-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
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
#include "ompi/mpiext/pcollreq/c/mpiext_pcollreq_c.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#    if OPAL_HAVE_WEAK_SYMBOLS
#        pragma weak MPIX_Palltoallv_init = PMPIX_Palltoallv_init
#    endif
#    define MPIX_Palltoallv_init PMPIX_Palltoallv_init
#endif

static const char FUNC_NAME[] = "MPIX_Palltoallv_init";

int MPIX_Palltoallv_init(const void *sendbuf,
                         const int sendpartitions[],
                         const int sendcounts[],
                         const int sdispls[],
                         MPI_Datatype sendtype,
                         void *recvbuf,
                         const int recvpartitions[],
                         const int recvcounts[],
                         const int rdispls[],
                         MPI_Datatype recvtype,
                         MPI_Comm comm,
                         MPI_Info info,
                         MPI_Request *request)
{
    int i, size, err;

    SPC_RECORD(OMPI_SPC_ALLTOALLV_INIT, 1);

    MEMCHECKER(
        ptrdiff_t recv_ext;
        ptrdiff_t send_ext;

        memchecker_comm(comm);

        if (MPI_IN_PLACE != sendbuf) {
            memchecker_datatype(sendtype);
            ompi_datatype_type_extent(sendtype, &send_ext);
        }

        memchecker_datatype(recvtype);
        ompi_datatype_type_extent(recvtype, &recv_ext);

        size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
        for (i = 0; i < size; i++) {
            if (MPI_IN_PLACE != sendbuf) {
                /* check if send chunks are defined. */
                memchecker_call(&opal_memchecker_base_isdefined,
                                (char *) (sendbuf) + sdispls[i] * send_ext,
                                sendpartitions[i] * sendcounts[i],
                                sendtype);
            }
            /* check if receive chunks are addressable. */
            memchecker_call(&opal_memchecker_base_isaddressable,
                            (char *) (recvbuf) + rdispls[i] * recv_ext,
                            recvpartitions[i] * recvcounts[i],
                            recvtype);
        });

    if (MPI_PARAM_CHECK) {

        /* Unrooted operation -- same checks for all ranks */

        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM, FUNC_NAME);
        }

        if (MPI_IN_PLACE == sendbuf) {
            sendpartitions = recvpartitions;
            sendcounts = recvcounts;
            sdispls = rdispls;
            sendtype = recvtype;
        }

        if ((NULL == sendpartitions) || (NULL == sendcounts) || (NULL == sdispls) ||
            (NULL == recvpartitions) || (NULL == recvcounts) || (NULL == rdispls)
            || (MPI_IN_PLACE == sendbuf && OMPI_COMM_IS_INTER(comm)) || MPI_IN_PLACE == recvbuf) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        }

        size = OMPI_COMM_IS_INTER(comm) ? ompi_comm_remote_size(comm) : ompi_comm_size(comm);
        for (i = 0; i < size; ++i) {
            OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendpartitions[i] * sendcounts[i]);
            OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
            OMPI_CHECK_DATATYPE_FOR_RECV(err, recvtype, recvpartitions[i] * recvcounts[i]);
            OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
        }

        if (MPI_IN_PLACE != sendbuf && !OMPI_COMM_IS_INTER(comm)) {
            int me = ompi_comm_rank(comm);
            size_t sendtype_size, recvtype_size;
            size_t send_size, recv_size;
            ompi_datatype_type_size(sendtype, &sendtype_size);
            ompi_datatype_type_size(recvtype, &recvtype_size);
            send_size = sendtype_size * sendpartitions[me] * sendcounts[me];
            recv_size = recvtype_size * recvpartitions[me] * recvcounts[me];
            if (send_size != recv_size) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TRUNCATE, FUNC_NAME);
            }
        }
    }

    if (OPAL_UNLIKELY(NULL == comm->c_coll->coll_palltoallv_init)) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_UNSUPPORTED_OPERATION, FUNC_NAME);
    }
    /* Invoke the coll component to perform the back-end operation */
    err = comm->c_coll->coll_palltoallv_init(sendbuf, sendpartitions, sendcounts, sdispls, sendtype,
                                             recvbuf, recvpartitions, recvcounts, rdispls, recvtype,
                                             comm, info, request, comm->c_coll->coll_palltoallv_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_datatypes(*request,
					 (MPI_IN_PLACE == sendbuf) ? NULL : sendtype,
                                        recvtype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
