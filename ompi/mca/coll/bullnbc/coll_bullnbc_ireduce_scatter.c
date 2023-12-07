/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      The University of Tennessee and The University
 *                         of Tennessee Research Foundation. All rights
 *                         reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "opal/align.h"
#include "opal_config.h"
#include "opal/util/bit_ops.h"
#include "opal/runtime/opal_params.h"

#include "coll_bullnbc_internal.h"

enum alg {
    IRED_SCAT_LINEAR = 1,
    IRED_SCAT_RECURSIVE_HALVING,
    IRED_SCAT_RING,
    IRED_SCAT_NB_ALG
};

static mca_base_var_enum_value_t reduce_scatter_algorithms[] = {
    {0, "ignore"},
    {IRED_SCAT_LINEAR, "basic"},
    {IRED_SCAT_RECURSIVE_HALVING, "recursive_halving"},
    {IRED_SCAT_RING, "ring"},
    {0, NULL}
};

/*
 *  reduce_scatter_intra_basic_recursivehalving
 *
 *  Function:   - reduce scatter implementation using recursive-halving
 *                algorithm
 *  Accepts:    - same as MPI_Reduce_scatter()
 *  Returns:    - MPI_SUCCESS or error code
 *  Limitation: - Works only for commutative operations.
 */
static int
bullnbc_reduce_scatter_recursivehalving( const void *sbuf, void *rbuf, const int *rcounts,
                                         struct ompi_datatype_t *dtype,
                                         struct ompi_op_t *op,
                                         struct ompi_communicator_t *comm,
                                         ompi_request_t ** request,
                                         mca_coll_base_module_t *module,
                                         bool persistent) {
    BULLNBC_Schedule *schedule;
    int i, rank, size, count, err;
    int tmp_size, remain = 0, tmp_rank, *disps;
    ptrdiff_t extent, buf_size, gap = 0;
    char *recv_buf = NULL, *recv_buf_free = NULL;
    char *result_buf = NULL, *result_buf_free = NULL;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    /* Initialize */
    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm);

    /* Find displacements and the like */
    disps = (int*) malloc(sizeof(int) * size);
    if (NULL == disps) return OMPI_ERR_OUT_OF_RESOURCE;

    disps[0] = 0;
    for (i = 0; i < (size - 1); ++i) {
        disps[i + 1] = disps[i] + rcounts[i];
    }
    count = disps[size - 1] + rcounts[size - 1];

    /* short cut the trivial case */
    if (0 == count) {
        free(disps);
        return OMPI_SUCCESS;
    }

    /* get datatype information */
    ompi_datatype_type_extent(dtype, &extent);
    buf_size = opal_datatype_span(&dtype->super, count, &gap);

    /* Handle MPI_IN_PLACE */
    if (MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);

    /* Allocate temporary receive buffer. */
    recv_buf_free = (char*) malloc( 2* buf_size);
    if (NULL == recv_buf_free) {
        err = OMPI_ERR_OUT_OF_RESOURCE;
        goto cleanup;
    }
    recv_buf = recv_buf_free - gap;

    result_buf_free = (char*)recv_buf_free + buf_size;
    result_buf = result_buf_free - gap;

    /* copy local buffer into the temporary results */
    err = ompi_datatype_sndrcv(sbuf, count, dtype, result_buf, count, dtype);
    if (OMPI_SUCCESS != err) goto cleanup;

    /* figure out power of two mapping: grow until larger than
       comm size, then go back one, to get the largest power of
       two less than comm size */
    tmp_size = opal_next_poweroftwo (size);
    tmp_size >>= 1;
    if (OPAL_UNLIKELY(tmp_size <= 0)) {
        /* size >= 1 so next power return at least 2, so tmp_size > 0
         * but sonarqube doubts around line 200 if "tmp_size-1 >= 0" :'( */
        NBC_DEBUG(1, "%s called with comm_size %d that is reduced to %d"
                  " causing underflow in for loops",
                  __func__, size, tmp_size);
        err = OMPI_ERR_BAD_PARAM;
        goto cleanup;
    }
    remain = size - tmp_size;

    /* If comm size is not a power of two, have the first "remain"
       procs with an even rank send to rank + 1, leaving a power of
       two procs to do the rest of the algorithm */
    if (rank < 2 * remain) {
        if ((rank & 1) == 0) {
            err = BULLNBC_Sched_send(result_buf, false, count, dtype, rank + 1,
                                 schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
                goto cleanup;
            }
            /* we don't participate from here on out */
            tmp_rank = -1;
        } else {
            err = BULLNBC_Sched_recv(recv_buf, false, count, dtype, rank - 1,
                                            schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
                goto cleanup;
            }

            /* integrate their results into our temp results */
            /* FIXME */
            BULLNBC_Sched_op(recv_buf, false, result_buf, false, count, dtype, op, schedule, true);

            /* adjust rank to be the bottom "remain" ranks */
            tmp_rank = rank / 2;
        }
    } else {
        /* just need to adjust rank to show that the bottom "even
           remain" ranks dropped out */
        tmp_rank = rank - remain;
    }

    /* For ranks not kicked out by the above code, perform the
       recursive halving */
    if (tmp_rank >= 0) {
        int *tmp_disps = NULL, *tmp_rcounts;
        int mask, send_index, recv_index, last_index;

        /* recalculate disps and rcounts to account for the
           special "remainder" processes that are no longer doing
           anything */
        tmp_rcounts = (int*) malloc(tmp_size * sizeof(int));
        if (NULL == tmp_rcounts) {
            err = OMPI_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }
        tmp_disps = (int*) malloc(tmp_size * sizeof(int));
        if (NULL == tmp_disps) {
            free(tmp_rcounts);
            err = OMPI_ERR_OUT_OF_RESOURCE;
            goto cleanup;
        }

        for (i = 0 ; i < tmp_size ; ++i) {
            if (i < remain) {
                /* need to include old neighbor as well */
                tmp_rcounts[i] = rcounts[i * 2 + 1] + rcounts[i * 2];
            } else {
                tmp_rcounts[i] = rcounts[i + remain];
            }
        }

        tmp_disps[0] = 0;
        for (i = 0; i < tmp_size - 1; ++i) {
            tmp_disps[i + 1] = tmp_disps[i] + tmp_rcounts[i];
        }

        /* do the recursive halving communication.  Don't use the
           dimension information on the communicator because I
           think the information is invalidated by our "shrinking"
           of the communicator */
        mask = tmp_size >> 1;
        send_index = recv_index = 0;
        last_index = tmp_size;
        while (mask > 0) {
            int tmp_peer, peer, send_count, recv_count;

            tmp_peer = tmp_rank ^ mask;
            peer = (tmp_peer < remain) ? tmp_peer * 2 + 1 : tmp_peer + remain;

            /* figure out if we're sending, receiving, or both */
            send_count = recv_count = 0;
            if (tmp_rank < tmp_peer) {
                send_index = recv_index + mask;
                for (i = send_index ; i < last_index ; ++i) {
                    send_count += tmp_rcounts[i];
                }
                for (i = recv_index ; i < send_index ; ++i) {
                    recv_count += tmp_rcounts[i];
                }
            } else {
                recv_index = send_index + mask;
                for (i = send_index ; i < recv_index ; ++i) {
                    send_count += tmp_rcounts[i];
                }
                for (i = recv_index ; i < last_index ; ++i) {
                    recv_count += tmp_rcounts[i];
                }
            }

            /* actual data transfer.  Send from result_buf,
               receive into recv_buf */
            if (send_count > 0) {
                //fprintf(stdout, "send %d(+%d) data to #%d\n", send_count,tmp_disps[send_index], peer); fflush(stdout);
                err = BULLNBC_Sched_send(result_buf + (ptrdiff_t)tmp_disps[send_index] * extent, false,
                                        send_count, dtype, peer,
                                        schedule, false);
                if (OMPI_SUCCESS != err) {
                    free(tmp_rcounts);
                    free(tmp_disps);
                    goto cleanup;
                }
            }
            if (recv_count > 0) {
                //fprintf(stdout, "recv %d+%d data from %d\n", recv_count,tmp_disps[recv_index],  peer); fflush(stdout);
                err = BULLNBC_Sched_recv(recv_buf + (ptrdiff_t)tmp_disps[recv_index] * extent, false,
                                         recv_count, dtype, peer,
                                         schedule, true);
                if (OMPI_SUCCESS != err) {
                    free(tmp_rcounts);
                    free(tmp_disps);
                    goto cleanup;
                }

                /* if we received something on this step, push it into
                   the results buffer */
                err = BULLNBC_Sched_op(recv_buf + (ptrdiff_t)tmp_disps[recv_index] * extent,
                               false,
                               result_buf + (ptrdiff_t)tmp_disps[recv_index] * extent,
                               false,
                               recv_count, dtype, op, schedule, true);
                if (OMPI_SUCCESS != err) {
                    free(tmp_rcounts);
                    free(tmp_disps);
                    goto cleanup;
                }
            }

            /* update for next iteration */
            send_index = recv_index;
            last_index = recv_index + mask;
            mask >>= 1;
        }

        /* copy local results from results buffer into real receive buffer */
        if (0 != rcounts[rank]) {
            //fprintf(stdout, "copy for me %d+%d data\n", rcounts[rank], disps[rank] ); fflush(stdout);
            err = NBC_Sched_copy(result_buf + disps[rank] * extent, false,
                                 rcounts[rank], dtype,
                                 rbuf, false, rcounts[rank], dtype,
                                 schedule, false);
            if (OMPI_SUCCESS != err) {
                free(tmp_rcounts);
                free(tmp_disps);
                goto cleanup;
            }
        }

        free(tmp_rcounts);
        free(tmp_disps);
        tmp_rcounts = tmp_disps = NULL;
    }

    /* Now fix up the non-power of two case, by having the odd
       procs send the even procs the proper results */
    if (rank < (2 * remain)) {
        if ((rank & 1) == 0) {
            if (rcounts[rank]) {
                //fprintf(stdout, "recv %d data from %d\n", rcounts[rank], rank + 1); fflush(stdout);
                err = BULLNBC_Sched_recv(rbuf, false, rcounts[rank], dtype, rank + 1,
                            schedule, true);
                if (OMPI_SUCCESS != err) goto cleanup;
            }
        } else {
            if (rcounts[rank - 1]) {
                //fprintf(stdout, "send %d(+%d) data to %d\n", rcounts[rank - 1],disps[rank - 1],rank - 1); fflush(stdout);
                err = BULLNBC_Sched_send(result_buf + disps[rank - 1] * extent, false,
                                        rcounts[rank - 1], dtype, rank - 1,
                                        schedule, false);
                if (OMPI_SUCCESS != err) goto cleanup;
            }
        }
    }

    err = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        goto cleanup;
    }

    err = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent,
                               request, recv_buf_free);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        goto cleanup;
    }
    free(disps);
    return OMPI_SUCCESS;

cleanup:
    OBJ_RELEASE(schedule);
    free(recv_buf_free);
    free(disps);
    return err;
}

/*
 *   ompi_coll_base_reduce_scatter_intra_ring
 *
 *   Function:       Ring algorithm for reduce_scatter operation
 *   Accepts:        Same as MPI_Reduce_scatter()
 *   Returns:        MPI_SUCCESS or error code
 *
 *   Description:    Implements ring algorithm for reduce_scatter:
 *                   the block sizes defined in rcounts are exchanged and
 8                    updated until they reach proper destination.
 *                   Algorithm requires 2 * max(rcounts) extra buffering
 *
 *   Limitations:    The algorithm DOES NOT preserve order of operations so it
 *                   can be used only for commutative operations.
 *         Example on 5 nodes:
 *         Initial state
 *   #      0              1             2              3             4
 *        [00]           [10]   ->     [20]           [30]           [40]
 *        [01]           [11]          [21]  ->       [31]           [41]
 *        [02]           [12]          [22]           [32]  ->       [42]
 *    ->  [03]           [13]          [23]           [33]           [43] --> ..
 *        [04]  ->       [14]          [24]           [34]           [44]
 *
 *        COMPUTATION PHASE
 *         Step 0: rank r sends block (r-1) to rank (r+1) and
 *                 receives block (r+1) from rank (r-1) [with wraparound].
 *   #      0              1             2              3             4
 *        [00]           [10]        [10+20]   ->     [30]           [40]
 *        [01]           [11]          [21]          [21+31]  ->     [41]
 *    ->  [02]           [12]          [22]           [32]         [32+42] -->..
 *      [43+03] ->       [13]          [23]           [33]           [43]
 *        [04]         [04+14]  ->     [24]           [34]           [44]
 *
 *         Step 1:
 *   #      0              1             2              3             4
 *        [00]           [10]        [10+20]       [10+20+30] ->     [40]
 *    ->  [01]           [11]          [21]          [21+31]      [21+31+41] ->
 *     [32+42+02] ->     [12]          [22]           [32]         [32+42]
 *        [03]        [43+03+13] ->    [23]           [33]           [43]
 *        [04]         [04+14]      [04+14+24]  ->    [34]           [44]
 *
 *         Step 2:
 *   #      0              1             2              3             4
 *     -> [00]           [10]        [10+20]       [10+20+30]   [10+20+30+40] ->
 *   [21+31+41+01]->     [11]          [21]          [21+31]      [21+31+41]
 *     [32+42+02]   [32+42+02+12]->    [22]           [32]         [32+42]
 *        [03]        [43+03+13]   [43+03+13+23]->    [33]           [43]
 *        [04]         [04+14]      [04+14+24]    [04+14+24+34] ->   [44]
 *
 *         Step 3:
 *   #      0             1              2              3             4
 * [10+20+30+40+00]     [10]         [10+20]       [10+20+30]   [10+20+30+40]
 *  [21+31+41+01] [21+31+41+01+11]     [21]          [21+31]      [21+31+41]
 *    [32+42+02]   [32+42+02+12] [32+42+02+12+22]     [32]         [32+42]
 *       [03]        [43+03+13]    [43+03+13+23] [43+03+13+23+33]    [43]
 *       [04]         [04+14]       [04+14+24]    [04+14+24+34] [04+14+24+34+44]
 *    DONE :)
 *
 */
static int
bullnbc_reduce_scatter_ring( const void *sbuf, void *rbuf, const int *rcounts,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op,
                             struct ompi_communicator_t *comm,
                             ompi_request_t ** request,
                             mca_coll_base_module_t *module,
                             bool persistent) {
    int ret, rank, size, i, k, recv_from, send_to, total_count, max_block_count;
    int inbi, *displs;
    char *tmpsend = NULL, *tmprecv = NULL, *accumbuf = NULL, *accumbuf_free = NULL;
    char *inbuf[2] = {NULL, NULL};
    ptrdiff_t extent, max_real_segsize, dsize, gap = 0;
    ompi_coll_bullnbc_module_t *bullnbc_module;
    BULLNBC_Schedule *schedule;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    schedule = OBJ_NEW(BULLNBC_Schedule);
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Determine the maximum number of elements per node,
       corresponding block size, and displacements array.
       */
    displs = (int*) malloc(size * sizeof(int));
    if (NULL == displs) { ret = OMPI_ERR_OUT_OF_RESOURCE; goto error_hndl; }

    displs[0] = 0;
    total_count = rcounts[0];
    max_block_count = rcounts[0];
    for (i = 1; i < size; i++) {
        displs[i] = total_count;
        total_count += rcounts[i];
        if (max_block_count < rcounts[i]){
            max_block_count = rcounts[i];
        }
    }

    /* Special case for size == 1 */
    if (1 == size) {
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_copy_content_same_ddt(dtype, total_count,
                                                      (char*)rbuf, (char*)sbuf);
            if (ret < 0) { goto error_hndl; }
        }
        free(displs);
        return MPI_SUCCESS;
    }

    /* Allocate and initialize temporary buffers, we need:
       - a temporary buffer to perform reduction (size total_count) since
       rbuf can be of rcounts[rank] size.
       - up to two temporary buffers used for communication/computation overlap.
    */
    ret = ompi_datatype_type_extent(dtype, &extent);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    max_real_segsize = opal_datatype_span(&dtype->super, max_block_count, &gap);
    dsize = opal_datatype_span(&dtype->super, total_count, &gap);

    accumbuf_free = (char*)malloc(dsize + 2*max_real_segsize );
    if (NULL == accumbuf_free) { ret = OMPI_ERR_OUT_OF_RESOURCE; goto error_hndl; }

    accumbuf = accumbuf_free - gap;
    inbuf[0] = accumbuf_free + dsize - gap;
    if (size > 2) {
        inbuf[1] = inbuf[0] + max_real_segsize;
    }

    /* Handle MPI_IN_PLACE for size > 1 */
    if (MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
    }

    ret = ompi_datatype_copy_content_same_ddt(dtype, total_count,
                                              accumbuf, (char*)sbuf);
    if (ret < 0) { goto error_hndl; }

    /* Computation loop */

    /*
       For each of the remote nodes:
       - post irecv for block (r-2) from (r-1) with wrap around
       - send block (r-1) to (r+1)
       - in loop for every step k = 2 .. n
       - post irecv for block (r - 1 + n - k) % n
       - wait on block (r + n - k) % n to arrive
       - compute on block (r + n - k ) % n
       - send block (r + n - k) % n
       - wait on block (r)
       - compute on block (r)
       - copy block (r) to rbuf
       Note that we must be careful when computing the begining of buffers and
       for send operations and computation we must compute the exact block size.
    */
    send_to = (rank + 1) % size;
    recv_from = (rank + size - 1) % size;

    inbi = 0;

    ret = BULLNBC_Sched_recv(inbuf[inbi], false, max_block_count, dtype, recv_from,
                                 schedule, false);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    tmpsend = accumbuf + (ptrdiff_t)displs[recv_from] * extent;
    ret = BULLNBC_Sched_send(tmpsend, false, rcounts[recv_from], dtype, send_to,
                             schedule, true);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    for (k = 2; k < size; k++) {
        const int prevblock = (rank + size - k) % size;
        inbi = inbi ^ 0x1;

        /* Apply operation on previous block: result goes to rbuf
           rbuf[prevblock] = inbuf[inbi ^ 0x1] (op) rbuf[prevblock]
        */
        tmprecv = accumbuf + (ptrdiff_t)displs[prevblock] * extent;
        ret = BULLNBC_Sched_op(inbuf[inbi ^ 0x1], false, tmprecv, false,
                               rcounts[prevblock], dtype, op, schedule, true);
        if (MPI_SUCCESS != ret) { goto error_hndl; }

        /* Post irecv for the current block */
        ret = BULLNBC_Sched_recv(inbuf[inbi], false, max_block_count, dtype, recv_from,
                                 schedule, false);
        if (MPI_SUCCESS != ret) { goto error_hndl; }

        /* send previous block to send_to */
        ret = BULLNBC_Sched_send(tmprecv, false, rcounts[prevblock], dtype, send_to,
                                 schedule, true);
        if (MPI_SUCCESS != ret) { goto error_hndl; }
    }

    /* Apply operation on the last block (my block)
       rbuf[rank] = inbuf[inbi] (op) rbuf[rank] */
    tmprecv = accumbuf + (ptrdiff_t)displs[rank] * extent;
    ret = BULLNBC_Sched_op(inbuf[inbi], false, tmprecv, false,
                           rcounts[rank], dtype, op, schedule, true);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    /* Copy result from tmprecv to rbuf */
    ret = NBC_Sched_copy(tmprecv, false, rcounts[rank], dtype,
                         rbuf, false, rcounts[rank], dtype,
                         schedule, false);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    ret = NBC_Sched_commit (schedule);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    ret = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent,
                request, accumbuf_free);
    if (MPI_SUCCESS != ret) { goto error_hndl; }

    free(displs);
    return MPI_SUCCESS;

error_hndl:
    free(displs);
    free(accumbuf_free);
    OBJ_RELEASE(schedule);
    return ret;
}

/* binomial reduce to rank 0 followed by a linear scatter ...
 *
 * Algorithm:
 * binomial reduction
 * round r:
 *  grp = rank % 2^r
 *  if grp == 0: receive from rank + 2^(r-1) if it exists and reduce value
 *  if grp == 1: send to rank - 2^(r-1) and exit function
 *
 * do this for R=log_2(p) rounds
 *
 * Linear scatter
 *
 */

static int
bullnbc_reduce_scatter_default( //NOSONAR cognitive complexity
                               const void *sendbuf, void *recvbuf, const int *recvcounts,
                               struct ompi_datatype_t *datatype,
                               struct ompi_op_t *op,
                               struct ompi_communicator_t *comm,
                               ompi_request_t ** request,
                               mca_coll_base_module_t *module,
                               bool persistent) {
    int peer, rank, maxr, p, res, count;
    MPI_Aint ext;
    ptrdiff_t gap, span, span_align;
    char inplace;
    void *tmpbuf;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
    char *rbuf, *lbuf, *buf;
    BULLNBC_Schedule *schedule;



    NBC_IN_PLACE(sendbuf, recvbuf, inplace);

    rank = ompi_comm_rank (comm);
    p = ompi_comm_size (comm);

    res = ompi_datatype_type_extent (datatype, &ext);
    if (MPI_SUCCESS != res) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
        return res;
    }

    count = 0;
    for (int r = 0 ; r < p ; ++r) {
        count += recvcounts[r];
    }

    if ((1 == p && (!persistent || inplace)) || 0 == count) {
        if (!inplace) {
            /* single node not in_place: copy data to recvbuf */
            res = NBC_Copy(sendbuf, recvcounts[0], datatype, recvbuf, recvcounts[0], datatype, comm);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }

        return bullnbc_get_noop_request(persistent, request);
    }

    maxr = ceil_of_log2(p);

    span = opal_datatype_span(&datatype->super, count, &gap);
    span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
    tmpbuf = malloc (span_align + span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    rbuf = (char *)(-gap);
    lbuf = (char *)(span_align - gap);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        free(tmpbuf);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* Leaf are the only ranks that do not reduce data */
    char is_leaf = (p == rank+1 || rank % 2);
    int firstred = 1;
    uint64_t flags;
    int dev_id;
    if (!is_leaf && (opal_ireduce_scatter_use_device_pointers ||
                opal_accelerator.check_addr(sendbuf, &dev_id, &flags))) {
        /* First sched on sendbuf is OP, that needs a buffer on host */
        res = NBC_Sched_copy(sendbuf, false, count, datatype, lbuf, true, count, datatype, schedule,
                             true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
        firstred =0;
    }


    for (int r = 1; r <= maxr ; ++r) {
        if ((rank % (1 << r)) == 0) {
            /* we have to receive this round */
            peer = rank + (1 << (r - 1));
            if (peer < p) {
                /* we have to wait until we have the data */
                res = NBC_Sched_recv(rbuf, true, count, datatype, peer, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

                /* this cannot be done until tmpbuf is unused :-( so barrier after the op */
                if (firstred) {
                    /* take reduce data from the sendbuf in the first round -> save copy */
                    res = NBC_Sched_op (sendbuf, false, rbuf, true, count, datatype, op, schedule, true);
                    firstred = 0;
                } else {
                    /* perform the reduce in my local buffer */
                    res = NBC_Sched_op (lbuf, true, rbuf, true, count, datatype, op, schedule, true);
                }

                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
                /* swap left and right buffers */
                buf = rbuf; rbuf = lbuf ; lbuf = buf;
            }
        } else {
            /* we have to send this round */
            peer = rank - (1 << (r - 1));
            if (firstred) {
                /* we have to send the senbuf */
                res = NBC_Sched_send (sendbuf, false, count, datatype, peer, schedule, false);
            } else {
                /* we send an already reduced value from lbuf */
                res = NBC_Sched_send (lbuf, true, count, datatype, peer, schedule, false);
            }
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

            /* leave the game */
            break;
        }
    }

    res = NBC_Sched_barrier(schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

    /* rank 0 is root and sends - all others receive */
    if (rank == 0) {
        for (long int r = 1, offset = 0 ; r < p ; ++r) {
            char* sbuf;
            offset += recvcounts[r-1];
            sbuf = lbuf + (offset*ext);
            /* root sends the right buffer to the right receiver */
            res = NBC_Sched_send (sbuf, true, recvcounts[r], datatype, r, schedule,
                                  false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
        }

        if (p == 1) {
            /* single node not in_place: copy data to recvbuf */
            res = NBC_Sched_copy ((void *)sendbuf, false, recvcounts[0], datatype,
                                  recvbuf, false, recvcounts[0], datatype, schedule, false);
        } else {
            res = NBC_Sched_copy (lbuf, true, recvcounts[0], datatype, recvbuf, false,
                                  recvcounts[0], datatype, schedule, false);
        }
    } else {
        res = NBC_Sched_recv (recvbuf, false, recvcounts[rank], datatype, 0, schedule, false);
    }

    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

    res = NBC_Sched_commit (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

    res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, tmpbuf);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

    return OMPI_SUCCESS;

cleanup_and_fail:
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
}

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_reduce_scatter_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
    mca_base_var_enum_t *new_enum;
    int cnt;

    for( cnt = 0; NULL != reduce_scatter_algorithms[cnt].string; cnt++ );
    mca_param_indices->algorithm_count = cnt;

    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "ireduce_scatter_algorithm_count",
                                           "Number of reduce_scatter algorithms available",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_CONSTANT,
                                           &mca_param_indices->algorithm_count);

    mca_param_indices->algorithm = 0;
    (void) mca_base_var_enum_create("coll_bullnbc_reduce_scatter_algorithms", reduce_scatter_algorithms, &new_enum);
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "ireduce_scatter_algorithm",
                                           "Which reduce_scatter algorithm is used.",
                                           MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_ALL,
                                           &mca_param_indices->algorithm);
    mca_param_indices->segsize = 0;
    mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                    "ireduce_scatter_algorithm_segmentsize",
                                    "Segment size in bytes used by default for ireduce_scatter algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                    MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_ALL,
                                    &mca_param_indices->segsize);
    OBJ_RELEASE(new_enum);
    return OMPI_SUCCESS;
}

static int
bullnbc_reduce_scatter_init (const void* sendbuf, void* recvbuf,
                             const int *recvcounts,
                             MPI_Datatype datatype, MPI_Op op,
                             struct ompi_communicator_t *comm,
                             ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module,
                             bool persistent) {

    enum alg alg = IRED_SCAT_RECURSIVE_HALVING;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
    int res, size;
    size_t dsize;

    res = ompi_datatype_type_size (datatype, &dsize);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    size = ompi_comm_size (comm);

    int total_size = recvcounts[0];
    for (int i = 1; i < size; ++i) {
        total_size += recvcounts[i];
    }
    total_size *= dsize;

    if(mca_coll_bullnbc_component.use_dynamic_rules) {
        if(0 != mca_coll_bullnbc_component.forced_params[REDUCESCATTER].algorithm) {
            /* if op is not commutative or MPI_IN_PLACE was specified we have to deal with it */
            alg = mca_coll_bullnbc_component.forced_params[REDUCESCATTER].algorithm;
            goto selected_rule;
        }
        if(bullnbc_module->com_rules[REDUCESCATTER]) {
            int algorithm,dummy1,dummy2,dummy3;
            algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[REDUCESCATTER],
                                                                 total_size, &dummy1, &dummy2, &dummy3);
            if(algorithm) {
                alg = algorithm;
                goto selected_rule;
            }
        }
    }

    if (total_size < 4096){
        alg = IRED_SCAT_RECURSIVE_HALVING;
    } else {
        alg = IRED_SCAT_LINEAR;
    }

selected_rule:
    switch(alg){
        case IRED_SCAT_LINEAR:
            res = bullnbc_reduce_scatter_default(sendbuf, recvbuf, recvcounts, datatype, op,
                                                 comm, request, module, persistent);
            break;
        case IRED_SCAT_RECURSIVE_HALVING:
            res = bullnbc_reduce_scatter_recursivehalving(sendbuf, recvbuf, recvcounts, datatype, op,
                                                          comm, request, module, persistent);
            break;
        case IRED_SCAT_RING:
            res = bullnbc_reduce_scatter_ring(sendbuf, recvbuf, recvcounts, datatype, op,
                                              comm, request, module, persistent);
            break;
        default:
            NBC_Error("ERROR Line %d\n", __LINE__);
            return OMPI_ERROR;
    }
    if (OPAL_LIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

static int nbc_reduce_scatter_inter_init (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                          MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, res, count, lsize, rsize;
  MPI_Aint ext;
  ptrdiff_t gap, span, span_align;
  BULLNBC_Schedule *schedule;
  void *tmpbuf = NULL;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  lsize = ompi_comm_size(comm);
  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent (datatype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  count = 0;
  for (int r = 0 ; r < lsize ; ++r) {
    count += recvcounts[r];
  }

  span = opal_datatype_span(&datatype->super, count, &gap);
  span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);

  if (count > 0) {
    tmpbuf = malloc (span_align + span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* send my data to the remote root */
  res = NBC_Sched_send(sendbuf, false, count, datatype, 0, schedule, false);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  if (0 == rank) {
    char *lbuf, *rbuf;
    lbuf = (char *)(-gap);
    rbuf = (char *)(span_align-gap);
    res = NBC_Sched_recv (lbuf, true, count, datatype, 0, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    for (int peer = 1 ; peer < rsize ; ++peer) {
      char *tbuf;
      res = NBC_Sched_recv (rbuf, true, count, datatype, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      res = NBC_Sched_op (lbuf, true, rbuf, true, count, datatype,
                          op, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
      tbuf = lbuf; lbuf = rbuf; rbuf = tbuf;
    }

    /* do the local scatterv with the local communicator */
    res = NBC_Sched_copy (lbuf, true, recvcounts[0], datatype, recvbuf, false,
                          recvcounts[0], datatype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
    for (int peer = 1, offset = recvcounts[0] * ext; peer < lsize ; ++peer) {
      res = NBC_Sched_local_send (lbuf + offset, true, recvcounts[peer], datatype, peer, schedule,
                                  false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      offset += recvcounts[peer] * ext;
    }
  } else {
    /* receive my block */
    res = NBC_Sched_local_recv (recvbuf, false, recvcounts[rank], datatype, 0, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  res = NBC_Sched_commit (schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  res = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent, request, tmpbuf);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_ireduce_scatter_inter (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                            MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                            struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_inter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                            comm, request, module, false);
    if (OPAL_LIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    res = NBC_Start(*(ompi_coll_bullnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        BULLNBC_Return_handle (*(ompi_coll_bullnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

/* Non-persistent version */
int ompi_coll_bullnbc_ireduce_scatter (const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                      MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module) {
    int res = bullnbc_reduce_scatter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                      comm, request, module, false);
    if (OPAL_LIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    res = NBC_Start(*(ompi_coll_bullnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        BULLNBC_Return_handle (*(ompi_coll_bullnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

/* Persistant version */
int ompi_coll_bullnbc_reduce_scatter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                         MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module) {
    int res = bullnbc_reduce_scatter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                      comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_reduce_scatter_inter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                               MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_inter_init(sendbuf, recvbuf, recvcounts, datatype, op,
                                            comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
