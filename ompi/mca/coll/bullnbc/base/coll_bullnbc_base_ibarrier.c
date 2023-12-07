/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All Rights
 *                         reserved.
 * Copyright (c) 2015-2016 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * Copyright (c) 2021-2023 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "opal/util/bit_ops.h"
#include "ompi/constants.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/pml/pml.h"

#include "ompi/mca/coll/bullnbc/coll_bullnbc_internal.h"
#include "ompi/mca/coll/bullnbc/base/coll_bullnbc_base.h"

/*
 * Barrier is ment to be a synchronous operation, as some BTLs can mark
 * a request done before its passed to the NIC and progress might not be made
 * elsewhere we cannot allow a process to exit the barrier until its last
 * [round of] sends are completed.
 *
 * It is last round of sends rather than 'last' individual send as each pair of
 * peers can use different channels/devices/btls and the receiver of one of
 * these sends might be forced to wait as the sender
 * leaves the collective and does not make progress until the next mpi call
 *
 */

/*
 * Simple double ring version of barrier
 *
 * synchronous gurantee made by last ring of sends are synchronous
 *
 */
int
ompi_coll_bullnbc_base_ibarrier_intra_doublering(struct ompi_communicator_t *comm,
                                                 ompi_request_t ** request,
                                                 struct mca_coll_base_module_2_4_0_t *module,
                                                 bool persistent)
{
    int res;
    BULLNBC_Schedule *schedule;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    const int rank  = ompi_comm_rank(comm);
    const int size  = ompi_comm_size(comm);
    const int left  = ((rank - 1) % size);
    const int right = ((rank + 1) % size);

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (rank > 0) { /* receive message from the left */
        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, left, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    /* Send message to the right */
    res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, right, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    /* root needs to receive from the last node */
    if (rank == 0) {
        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, left, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    /* Do it twice, unroll version :) */

    /* Allow nodes to exit */
    if (rank > 0) { /* post Receive from left */
        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, left, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    /* send message to the right one */
    res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, right, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    /* root needs to receive from the last node */
    if (rank == 0) {
        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, left, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return OMPI_SUCCESS;
}


/*
 * To make synchronous, uses sync sends and sync sendrecvs
 */

int ompi_coll_bullnbc_base_ibarrier_intra_recursivedoubling(struct ompi_communicator_t *comm,
                                                           ompi_request_t ** request,
                                                           struct mca_coll_base_module_2_4_0_t *module,
                                                           bool persistent)
{
    BULLNBC_Schedule *schedule;
    int adjsize, res, remote;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    const int rank = ompi_comm_rank(comm);
    const int size = ompi_comm_size(comm);

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* do nearest power of 2 less than size calc */
    adjsize = opal_next_poweroftwo(size);
    adjsize >>= 1;

    /* if size is not exact power of two, perform an extra step */
    if (adjsize != size) {
        if (rank >= adjsize) {
            /* send message to lower ranked node */
            remote = rank - adjsize;

            res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = BULLNBC_Sched_barrier(schedule);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        } else if (rank < (size - adjsize)) {
            remote = rank + adjsize;

            res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = BULLNBC_Sched_barrier(schedule);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        }
    }

    /* exchange messages */
    if ( rank < adjsize ) {
        int mask = 0x1;

        while ( mask < adjsize ) {
            remote = rank ^ mask;
            mask <<= 1;

            if (remote >= adjsize) {
                continue;
            }

            /* post receive from the remote node */
            res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = BULLNBC_Sched_barrier(schedule);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        }
    }

    /* non-power of 2 case */
    if (adjsize != size) {
        if (rank < (size - adjsize)) {
            /* send enter message to higher ranked node */
            remote = rank + adjsize;

            res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }

            res = BULLNBC_Sched_barrier(schedule);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return MPI_SUCCESS;
}


/*
 * To make synchronous, uses sync sends and sync sendrecvs
 */

int ompi_coll_bullnbc_base_ibarrier_intra_bruck(struct ompi_communicator_t *comm,
                                                ompi_request_t ** request,
                                                struct mca_coll_base_module_2_4_0_t *module,
                                                bool persistent)
{
    int res;
    BULLNBC_Schedule *schedule;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    const int rank = ompi_comm_rank(comm);
    const int size = ompi_comm_size(comm);

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* exchange data with rank-2^k and rank+2^k */
    for (int distance = 1; distance < size; distance <<= 1) {
        const int from = (rank + size - distance) % size;
        const int to   = (rank + distance) % size;

        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, from, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }

        res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, to, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }

        res = BULLNBC_Sched_barrier(schedule);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return MPI_SUCCESS;
}


/*
 * To make synchronous, uses sync sends and sync sendrecvs
 */
/* special case for two processes */
int ompi_coll_bullnbc_base_ibarrier_intra_two_procs(struct ompi_communicator_t *comm,
                                                    ompi_request_t ** request,
                                                    struct mca_coll_base_module_2_4_0_t *module,
                                                    bool persistent)
{
    int remote, res;
    BULLNBC_Schedule *schedule;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    remote = ompi_comm_rank(comm);

    if (2 != ompi_comm_size(comm)) {
        return MPI_ERR_UNSUPPORTED_OPERATION;
    }

    remote = (remote + 1) & 0x1;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, remote, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, remote, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return res;
}


/*
 * Linear functions are copied from the BASIC coll module
 * they do not segment the message and are simple implementations
 * but for some small number of nodes and/or small data sizes they
 * are just as fast as base/tree based segmenting operations
 * and as such may be selected by the decision functions
 * These are copied into this module due to the way we select modules
 * in V1. i.e. in V2 we will handle this differently and so will not
 * have to duplicate code.
 * GEF Oct05 after asking Jeff.
 */

/* copied function (with appropriate renaming) starts here */
int ompi_coll_bullnbc_base_ibarrier_intra_basic_linear(struct ompi_communicator_t *comm,
                                                       ompi_request_t ** request,
                                                       struct mca_coll_base_module_2_4_0_t *module,
                                                       bool persistent)
{
    int res;
    BULLNBC_Schedule *schedule;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    const int rank = ompi_comm_rank(comm);
    const int size = ompi_comm_size(comm);

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* All non-root send & receive zero-length message.
     * The root collects and broadcasts the messages. */
    if (rank > 0) {

        res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, 0, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }

        res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, 0, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    } else {

        for (int i = 1; i < size; ++i) {
            res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, i, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        }

        res = BULLNBC_Sched_barrier(schedule);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }

        for (int i = 1; i < size; ++i) {
            res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, i, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                OBJ_RELEASE(schedule);
                return res;
            }
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return MPI_SUCCESS;
}

/*
 * Another recursive doubling type algorithm, but in this case
 * we go up the tree and back down the tree.
 */
int ompi_coll_bullnbc_base_ibarrier_intra_tree(struct ompi_communicator_t *comm,
                                               ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module,
                                               bool persistent)
{
    BULLNBC_Schedule *schedule;
    int depth, res, jump, partner;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    const int rank = ompi_comm_rank(comm);
    const int size = ompi_comm_size(comm);

    /* Find the nearest power of 2 of the communicator size. */
    depth = opal_next_poweroftwo_inclusive(size);

    bullnbc_module = (ompi_coll_bullnbc_module_t *) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* botton up ! */
    for (jump = 1; jump < depth; jump <<= 1) {
        partner = rank ^ jump;
        if (!(partner & (jump - 1)) && partner < size) {
            if (partner > rank) {
                res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, partner, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                    OBJ_RELEASE(schedule);
                    return res;
                }
            } else if (partner < rank) {
                res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, partner, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                    OBJ_RELEASE(schedule);
                    return res;
                }
            }
        }
    }

    /* Top down ! */
    depth >>= 1;
    for (jump = depth; jump > 0; jump >>= 1) {
        partner = rank ^ jump;
        if (!(partner & (jump-1)) && partner < size) {
            if (partner > rank) {
                res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, partner, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                    OBJ_RELEASE(schedule);
                    return res;
                }
            } else if (partner < rank) {
                res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, partner, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                    OBJ_RELEASE(schedule);
                    return res;
                }
            }
        }
    }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }

    return MPI_SUCCESS;
}
