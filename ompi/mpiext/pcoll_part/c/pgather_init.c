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
 * Copyright (c) 2006-2007 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2008      University of Houston.  All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
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
#include "ompi/mpiext/pcoll_part/c/mpiext_pcoll_part_c.h"
#include "ompi/runtime/ompi_spc.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Pgather_init = PMPIX_Pgather_init
#endif
#define MPIX_Pgather_init PMPIX_Pgather_init
#endif

static const char FUNC_NAME[] = "MPIX_Pgather_init";


int MPIX_Pgather_init(const void *sendbuf, int sendparts,
                      int sendcount, MPI_Datatype sendtype,
                      void *recvbuf, int recvparts,
                      int recvcount, MPI_Datatype recvtype,
                      int root, MPI_Comm comm, MPI_Info info, MPI_Request *request)
{
    int err;

    MEMCHECKER(
        // cppcheck-suppress unknownMacro
        int rank;
        ptrdiff_t ext;

        rank = ompi_comm_rank(comm);
        ompi_datatype_type_extent(recvtype, &ext);

        memchecker_comm(comm);
        if(OMPI_COMM_IS_INTRA(comm)) {
            if(ompi_comm_rank(comm) == root) {
                /* check whether root's send buffer is defined. */
                if (MPI_IN_PLACE == sendbuf) {
                  memchecker_call(&opal_memchecker_base_isdefined,
                                  (char *)(recvbuf)+rank*ext,
                                  recvcount, recvtype);
                } else {
                    memchecker_datatype(sendtype);
                    memchecker_call(&opal_memchecker_base_isdefined, sendbuf, sendcount, sendtype);
                }

                memchecker_datatype(recvtype);
                /* check whether root's receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, recvcount, recvtype);
            } else {
                memchecker_datatype(sendtype);
                /* check whether send buffer is defined on other processes. */
                memchecker_call(&opal_memchecker_base_isdefined, sendbuf, sendcount, sendtype);
            }
        } else {
            if (MPI_ROOT == root) {
                memchecker_datatype(recvtype);
                /* check whether root's receive buffer is addressable. */
                memchecker_call(&opal_memchecker_base_isaddressable, recvbuf, recvcount, recvtype);
            } else if (MPI_PROC_NULL != root) {
                memchecker_datatype(sendtype);
                /* check whether send buffer is defined. */
                memchecker_call(&opal_memchecker_base_isdefined, sendbuf, sendcount, sendtype);
            }
        }
    );

    if (MPI_PARAM_CHECK) {
        err = MPI_SUCCESS;
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);
        if (ompi_comm_invalid(comm)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_COMM,
                                          FUNC_NAME);
        } else if ((ompi_comm_rank(comm) != root && MPI_IN_PLACE == sendbuf) ||
                   (ompi_comm_rank(comm) == root && MPI_IN_PLACE == recvbuf)) {
            return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ARG, FUNC_NAME);
        }

        /* Errors for intracommunicators */

        if (OMPI_COMM_IS_INTRA(comm)) {

            /* Errors for all ranks */

            if ((root >= ompi_comm_size(comm)) || (root < 0)) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
            }

            if (MPI_IN_PLACE != sendbuf) {
                OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendcount);
            }
            OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);

            /* Errors for the root.  Some of these could have been
               combined into compound if statements above, but since
               this whole section can be compiled out (or turned off at
               run time) for efficiency, it's more clear to separate
               them out into individual tests. */

            if (ompi_comm_rank(comm) == root) {
                if (MPI_DATATYPE_NULL == recvtype || NULL == recvtype) {
                    return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TYPE, FUNC_NAME);
                }

                if (recvcount < 0) {
                    return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_COUNT, FUNC_NAME);
                }
            }
        }

        /* Errors for intercommunicators */

        else {
            if (! ((root >= 0 && root < ompi_comm_remote_size(comm)) ||
                   MPI_ROOT == root || MPI_PROC_NULL == root)) {
                return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_ROOT, FUNC_NAME);
            }

            /* Errors for the senders */

            if (MPI_ROOT != root && MPI_PROC_NULL != root) {
                OMPI_CHECK_DATATYPE_FOR_SEND(err, sendtype, sendcount);
                OMPI_ERRHANDLER_CHECK(err, comm, err, FUNC_NAME);
            }

            /* Errors for the root.  Ditto on the comment above -- these
               error checks could have been combined above, but let's
               make the code easier to read. */

            else if (MPI_ROOT == root) {
                if (recvcount < 0) {
                    return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_COUNT, FUNC_NAME);
                }

                if (MPI_DATATYPE_NULL == recvtype || NULL == recvtype) {
                    return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_TYPE, FUNC_NAME);
                }
            }
        }
    }

    /* Invoke the coll component to perform the back-end operation */
    if (OPAL_UNLIKELY(NULL == comm->c_coll->coll_pgather_init)) {
        return OMPI_ERRHANDLER_INVOKE(comm, MPI_ERR_UNSUPPORTED_OPERATION, FUNC_NAME);
    }

    err = comm->c_coll->coll_pgather_init(sendbuf, sendparts, sendcount, sendtype,
                                          recvbuf, recvparts, recvcount, recvtype,
                                          root, comm, info, request,
                                          comm->c_coll->coll_pgather_init_module);
    if (OPAL_LIKELY(OMPI_SUCCESS == err)) {
        if (OMPI_COMM_IS_INTRA(comm)) {
            if (MPI_IN_PLACE == sendbuf) {
                sendtype = NULL;
            } else if (ompi_comm_rank(comm) != root) {
                recvtype = NULL;
            }
        } else {
            if (MPI_ROOT == root) {
                sendtype = NULL;
            } else if (MPI_PROC_NULL == root) {
                sendtype = NULL;
                recvtype = NULL;
            } else {
                recvtype = NULL;
            }
        }
        ompi_coll_base_retain_datatypes(*request, sendtype, recvtype);
    }
    OMPI_ERRHANDLER_RETURN(err, comm, err, FUNC_NAME);
}
