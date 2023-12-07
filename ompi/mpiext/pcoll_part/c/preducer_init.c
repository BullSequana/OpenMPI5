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
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
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
#include "ompi/op/op.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/memchecker.h"
#include "ompi/mpiext/pcoll_part/c/mpiext_pcoll_part_c.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Preducer_init = PMPIX_Preducer_init
#endif
#define MPIX_Preducer_init PMPIX_Preducer_init
#endif

static const char FUNC_NAME[] = "MPIX_Preducer_init";


int MPIX_Preducer_init(const void *sendbuf, void *recvbuf, size_t parts, int count,
                       MPI_Datatype datatype,
                       MPI_Request * sreqs, MPI_Request * rreqs,
                       MPI_Op op, int root, MPI_Comm comm,
                       MPI_Info info, MPI_Request *request)
{
    int err;

    SPC_RECORD(OMPI_SPC_REDUCE_INIT, 1);

    MEMCHECKER(
        // cppcheck-suppress unknownMacro
        memchecker_datatype(datatype);
        memchecker_comm(comm);

        if(OMPI_COMM_IS_INTRA(comm)) {
            if(ompi_comm_rank(comm) == root) {
                /* check whether root's send buffer is defined. */
                if (MPI_IN_PLACE == sendbuf) {
                    memchecker_call(&opal_memchecker_base_isdefined, recvbuf, count, datatype);
                } else {
                    memchecker_call(&opal_memchecker_base_isdefined, sendbuf, count, datatype);
                }

                /* check whether root's receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, count, datatype);
            } else {
                /* check whether send buffer is defined on other processes. */
                memchecker_call(&opal_memchecker_base_isdefined, sendbuf, count, datatype);
            }
        } else {
            if (MPI_ROOT == root) {
                /* check whether root's receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, count, datatype);
            } else if (MPI_PROC_NULL != root) {
                /* check whether send buffer is defined. */
                memchecker_call(&opal_memchecker_base_isdefined, sendbuf, count, datatype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
        char *msg;
        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM,
                                          FUNC_NAME);
        }

        /* Checks for all ranks */

        else if (MPI_OP_NULL == op || NULL == op) {
            err = MPI_ERR_OP;
        } else if (!ompi_op_is_valid(op, datatype, &msg, FUNC_NAME)) {
            int ret = OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_OP, msg);
            free(msg);
            return ret;
        } else if ((ompi_comm_rank(comm) != root && ((MPI_IN_PLACE == sendbuf)
                                                     || (MPIX_NO_REQUESTS != rreqs))
                    ||
                   (ompi_comm_rank(comm) == root && ((MPI_IN_PLACE == recvbuf)
                                                     || (sendbuf == recvbuf))))) {
            err = MPI_ERR_ARG;
        } else {
            OMPI_CHECK_DATATYPE_FOR_SEND(err, datatype, count);
        }
        OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);

        /* Intercommunicator errors */

        if (!OMPI_COMM_IS_INTRA(comm)) {
            if (! ((root >= 0 && root < ompi_comm_remote_size(comm)) ||
                   MPI_ROOT == root || MPI_PROC_NULL == root)) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
            }
        }

        /* Intracommunicator errors */

        else {
            if (root < 0 || root >= ompi_comm_size(comm)) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
            }
        }
    }

    /* Invoke the coll component to perform the back-end operation */
    if (OPAL_UNLIKELY(NULL == comm->c_coll->coll_preducer_init)) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_UNSUPPORTED_OPERATION, FUNC_NAME);
    }
    err = comm->c_coll->coll_preducer_init(sendbuf, recvbuf, parts, count, datatype,
                                           sreqs, rreqs,
                                           op, root, comm, info, request,
                                           comm->c_coll->coll_preducer_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        ompi_coll_base_retain_op(*request, op, datatype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
