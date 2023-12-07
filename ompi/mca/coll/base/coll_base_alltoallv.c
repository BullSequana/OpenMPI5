/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Sun Microsystems, Inc.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All Rights
 *                         reserved.
 * Copyright (c) 2013      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * Copyright (c) 2021      Amazon.com, Inc. or its affiliates.  All Rights
 *                         reserved.
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_convertor_internal.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_base_topo.h"
#include "coll_base_util.h"
#include "opal/util/minmax.h"

/*
 * We want to minimize the amount of temporary memory needed while allowing as many ranks
 * to exchange data simultaneously. We use a variation of the ring algorithm, where in a
 * single step a process exchange the data with both neighbors at distance k (on the left
 * and the right on a logical ring topology). With this approach we need to pack the data
 * for a single of the two neighbors, as we can then use the original buffer (and datatype
 * and count) to send the data to the other.
 */
int
mca_coll_base_alltoallv_intra_basic_inplace(const void *rbuf, const int *rcounts, const int *rdisps,
                                            struct ompi_datatype_t *rdtype,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    int i, size, rank, left, right, err = MPI_SUCCESS, line;
    ptrdiff_t extent;
    ompi_request_t *req = MPI_REQUEST_NULL;
    char *tmp_buffer;
    size_t packed_size = 0, max_size, type_size;
    opal_convertor_t convertor;

    /* Initialize. */

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    ompi_datatype_type_size(rdtype, &type_size);

    for (i = 0, max_size = 0 ; i < size ; ++i) {
        if (i == rank) {
            continue;
        }
        packed_size = rcounts[i] * type_size;
        max_size = opal_max(packed_size, max_size);
    }

    /* Easy way out */
    if ((1 == size) || (0 == max_size) ) {
        return MPI_SUCCESS;
    }

    /* Find the largest amount of packed send/recv data among all peers where
     * we need to pack before the send.
     */
#if OPAL_ENABLE_HETEROGENEOUS_SUPPORT
    for (i = 1 ; i <= (size >> 1) ; ++i) {
        right = (rank + i) % size;
        ompi_proc_t *ompi_proc = ompi_comm_peer_lookup(comm, right);

        if( OPAL_UNLIKELY(opal_local_arch != ompi_proc->super.proc_convertor->master->remote_arch))  {
            packed_size = opal_datatype_compute_remote_size(&rdtype->super,
                                                            ompi_proc->super.proc_convertor->master->remote_sizes);
            packed_size *= rcounts[right];
            max_size = packed_size > max_size ? packed_size : max_size;
        }
    }
#endif  /* OPAL_ENABLE_HETEROGENEOUS_SUPPORT */

    ompi_datatype_type_extent(rdtype, &extent);

    /* Allocate a temporary buffer */
    tmp_buffer = calloc (max_size, 1);
    if( NULL == tmp_buffer) { err = OMPI_ERR_OUT_OF_RESOURCE; line = __LINE__; goto error_hndl; }

    for (i = 1 ; i <= (size >> 1) ; ++i) {
        struct iovec iov = {.iov_base = tmp_buffer, .iov_len = max_size};
        uint32_t iov_count = 1;

        right = (rank + i) % size;
        left  = (rank + size - i) % size;

        if( 0 != rcounts[right] ) {  /* nothing to exchange with the peer on the right */
            ompi_proc_t *right_proc = ompi_comm_peer_lookup(comm, right);
            opal_convertor_clone(right_proc->super.proc_convertor, &convertor, 0);
            opal_convertor_prepare_for_send(&convertor, &rdtype->super, rcounts[right],
                                            (char *) rbuf + rdisps[right] * extent);
            packed_size = max_size;
            err = opal_convertor_pack(&convertor, &iov, &iov_count, &packed_size);
            if (1 != err) {
                line = __LINE__;
                goto error_hndl;
            }

            /* Receive data from the right */
            err = MCA_PML_CALL(irecv ((char *) rbuf + rdisps[right] * extent, rcounts[right], rdtype,
                                      right, MCA_COLL_BASE_TAG_ALLTOALLV, comm, &req));
            if (MPI_SUCCESS != err) {
                line = __LINE__;
                goto error_hndl;
            }
        }

        if( (left != right) && (0 != rcounts[left]) ) {
            /* Send data to the left */
            err = MCA_PML_CALL(send ((char *) rbuf + rdisps[left] * extent, rcounts[left], rdtype,
                                     left, MCA_COLL_BASE_TAG_ALLTOALLV, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) {
                line = __LINE__;
                goto error_hndl;
             }

            err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
            if (MPI_SUCCESS != err) {
                line = __LINE__;
                goto error_hndl;
             }

            /* Receive data from the left */
            err = MCA_PML_CALL(irecv ((char *) rbuf + rdisps[left] * extent, rcounts[left], rdtype,
                                      left, MCA_COLL_BASE_TAG_ALLTOALLV, comm, &req));
            if (MPI_SUCCESS != err) {
                line = __LINE__;
                goto error_hndl;
            }
        }

        if( 0 != rcounts[right] ) {  /* nothing to exchange with the peer on the right */
            /* Send data to the right */
            err = MCA_PML_CALL(send ((char *) tmp_buffer,  packed_size, MPI_PACKED,
                                     right, MCA_COLL_BASE_TAG_ALLTOALLV, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) {
                line = __LINE__;
                goto error_hndl;
            }
        }

        err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
        if (MPI_SUCCESS != err) {
            line = __LINE__;
            goto error_hndl;
        }
    }

 error_hndl:
    /* Free the temporary buffer */
    if( NULL != tmp_buffer )
        free (tmp_buffer);

    if( MPI_SUCCESS != err ) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "%s:%4d\tError occurred %d, rank %2d", __FILE__, line, err, rank));
        (void)line;  // silence compiler warning
    }

    /* All done */
    return err;
}

int
ompi_coll_base_alltoallv_intra_pairwise(const void *sbuf, const int *scounts, const int *sdisps,
                                         struct ompi_datatype_t *sdtype,
                                         void* rbuf, const int *rcounts, const int *rdisps,
                                         struct ompi_datatype_t *rdtype,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    int line = -1, err = 0, rank, size, step = 0, sendto, recvfrom;
    void *psnd, *prcv;
    ptrdiff_t sext, rext;

    if (MPI_IN_PLACE == sbuf) {
        return mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                             rdtype, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:alltoallv_intra_pairwise rank %d", rank));

    ompi_datatype_type_extent(sdtype, &sext);
    ompi_datatype_type_extent(rdtype, &rext);

   /* Perform pairwise exchange starting from 1 since local exchange is done */
    for (step = 0; step < size; step++) {

        /* Determine sender and receiver for this step. */
        sendto  = (rank + step) % size;
        recvfrom = (rank + size - step) % size;

        /* Determine sending and receiving locations */
        psnd = (char*)sbuf + (ptrdiff_t)sdisps[sendto] * sext;
        prcv = (char*)rbuf + (ptrdiff_t)rdisps[recvfrom] * rext;

        /* send and receive */
        err = ompi_coll_base_sendrecv( psnd, scounts[sendto], sdtype, sendto,
                                        MCA_COLL_BASE_TAG_ALLTOALLV,
                                        prcv, rcounts[recvfrom], rdtype, recvfrom,
                                        MCA_COLL_BASE_TAG_ALLTOALLV,
                                        comm, MPI_STATUS_IGNORE, rank);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
    }

    return MPI_SUCCESS;

 err_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "%s:%4d\tError occurred %d, rank %2d at step %d", __FILE__, line,
                 err, rank, step));
    (void)line;  // silence compiler warning
    return err;
}

/* Algorithm similar to pairwise, but this one uses non blocking send/recv and
 * it detects if it is running in a sub comm, in this case a shift is created and
 * change the start step of the pairwise, each sub comm have different shift.
 * Nonblocking send/recv are limited to avoid performance loss.
 * This limit is controlled by a MCA parameter(OMPI_MCA_coll_tuned_alltoallv_pairwise_limit).*/
int
ompi_coll_base_alltoallv_intra_shifted_pairwise(const void *sbuf, const int *scounts, const int *sdisps,
                                                struct ompi_datatype_t *sdtype,
                                                void* rbuf, const int *rcounts, const int *rdisps,
                                                struct ompi_datatype_t *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module,
                                                int limit)
{
    int rank;
    int size;
    int step;
    int err;
    int line;
    int wsize;
    int start = 0;
    int index_s;
    int index_r;
    const void *psnd;
    void *prcv;
    ptrdiff_t sext;
    ptrdiff_t rext;
    struct ompi_request_t **send_rq;
    struct ompi_request_t **recv_rq;

    if (MPI_IN_PLACE == sbuf) {
        return mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                             rdtype, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Check if this collective is in a sub_comm */
    wsize = ompi_comm_size(MPI_COMM_WORLD); 
    if(size < wsize) {
        int tab_ranks=0;
        int tab_wranks;
        int nb_comm;
        int shift;
        ompi_communicator_t *world = MPI_COMM_WORLD;
        /* Search the first rank in each sub comm*/
        ompi_group_translate_ranks(comm->c_local_group, 1, &tab_ranks, world->c_local_group, &tab_wranks);
        nb_comm = wsize/size;
        shift = size/nb_comm;
        /* if shift is 0 we will necessarily have the same start */
        if (shift == 0) {
            shift = 1;
        }
        /* Check distribution and create different shift for each sub comm */
        if(tab_wranks % size != 0) {
            start = shift*(tab_wranks%nb_comm) % size;
        } else {
            start = shift*(tab_wranks/size) % size;
        }
    }
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:alltoallv_intra_shifted_pairwise rank %d", rank));

    ompi_datatype_type_extent(sdtype, &sext);
    ompi_datatype_type_extent(rdtype, &rext);
    if (limit == 0 || limit > size){
        limit = size;
    }
    recv_rq = malloc(limit * sizeof(struct ompi_request_t *));
    send_rq = malloc(limit * sizeof(struct ompi_request_t *));
    for (step = 0; step < size; step++) {
        int sendto;
        int recvfrom;
        /* Determine sender and receiver for this step. */
        sendto  = (rank + (step + start)) % size;
        recvfrom = (rank + size * 2 - (step + start)) % size;

        /* Determine sending and receiving locations */
        psnd = (const char*)sbuf + (ptrdiff_t)sdisps[sendto] * sext;
        prcv = (char*)rbuf + (ptrdiff_t)rdisps[recvfrom] * rext;

        /* Isend and Irecv until the limit is reached then wait until at least one of the messages is finished */
        if(step < limit) {
            err = MCA_PML_CALL(irecv(prcv, rcounts[recvfrom], rdtype,
                                     recvfrom, MCA_COLL_BASE_TAG_ALLTOALLV, comm,
                                     &recv_rq[step]));
            if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
            err = MCA_PML_CALL(isend(psnd, scounts[sendto], sdtype,
                                     sendto, MCA_COLL_BASE_TAG_ALLTOALLV,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     &send_rq[step]));
        } else {
            MPI_Waitany(limit, recv_rq, &index_r, MPI_STATUS_IGNORE);
            err = MCA_PML_CALL(irecv(prcv, rcounts[recvfrom], rdtype,
                                     recvfrom, MCA_COLL_BASE_TAG_ALLTOALLV, comm,
                                     recv_rq + index_r));
            if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
            MPI_Waitany(limit, send_rq, &index_s, MPI_STATUS_IGNORE);
            err = MCA_PML_CALL(isend(psnd, scounts[sendto], sdtype,
                                     sendto, MCA_COLL_BASE_TAG_ALLTOALLV,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     send_rq + index_s));
        }
        if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
    }

    /* Wait for completion */
    ompi_request_wait_all(limit, send_rq, MPI_STATUSES_IGNORE);
    ompi_request_wait_all(limit, recv_rq, MPI_STATUSES_IGNORE);
    free(send_rq);
    free(recv_rq);
    return MPI_SUCCESS;

 err_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "%s:%4d\tError occurred %d, rank %2d at step %d", __FILE__, line,
                 err, rank, step));
    (void)line;  // silence compiler warning
    return err;
}

/**
 * Linear functions are copied from the basic coll module.  For
 * some small number of nodes and/or small data sizes they are just as
 * fast as base/tree based segmenting operations and as such may be
 * selected by the decision functions.  These are copied into this module
 * due to the way we select modules in V1. i.e. in V2 we will handle this
 * differently and so will not have to duplicate code.
 */
int
ompi_coll_base_alltoallv_intra_basic_linear(const void *sbuf, const int *scounts, const int *sdisps,
                                            struct ompi_datatype_t *sdtype,
                                            void *rbuf, const int *rcounts, const int *rdisps,
                                            struct ompi_datatype_t *rdtype,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    int i, size, rank, err, nreqs;
    char *psnd, *prcv;
    ptrdiff_t sext, rext;
    ompi_request_t **preq, **reqs;
    mca_coll_base_module_t *base_module = (mca_coll_base_module_t*) module;
    mca_coll_base_comm_t *data = base_module->base_data;

    if (MPI_IN_PLACE == sbuf) {
        return  mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                              rdtype, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:alltoallv_intra_basic_linear rank %d", rank));

    ompi_datatype_type_extent(sdtype, &sext);
    ompi_datatype_type_extent(rdtype, &rext);

    /* Simple optimization - handle send to self first */
    psnd = ((char *) sbuf) + (ptrdiff_t)sdisps[rank] * sext;
    prcv = ((char *) rbuf) + (ptrdiff_t)rdisps[rank] * rext;
    if (0 != scounts[rank]) {
        err = ompi_datatype_sndrcv(psnd, scounts[rank], sdtype,
                              prcv, rcounts[rank], rdtype);
        if (MPI_SUCCESS != err) {
            return err;
        }
    }

    /* If only one process, we're done. */
    if (1 == size) {
        return MPI_SUCCESS;
    }

    /* Now, initiate all send/recv to/from others. */
    nreqs = 0;
    reqs = preq = ompi_coll_base_comm_get_reqs(data, 2 * size);
    if( NULL == reqs ) { err = OMPI_ERR_OUT_OF_RESOURCE; goto err_hndl; }

    /* Post all receives first */
    for (i = 0; i < size; ++i) {
        if (i == rank) {
            continue;
        }

        if (rcounts[i] > 0) {
            ++nreqs;
            prcv = ((char *) rbuf) + (ptrdiff_t)rdisps[i] * rext;
            err = MCA_PML_CALL(irecv_init(prcv, rcounts[i], rdtype,
                                          i, MCA_COLL_BASE_TAG_ALLTOALLV, comm,
                                          preq++));
            if (MPI_SUCCESS != err) { goto err_hndl; }
        }
    }

    /* Now post all sends */
    for (i = 0; i < size; ++i) {
        if (i == rank) {
            continue;
        }

        if (scounts[i] > 0) {
            ++nreqs;
            psnd = ((char *) sbuf) + (ptrdiff_t)sdisps[i] * sext;
            err = MCA_PML_CALL(isend_init(psnd, scounts[i], sdtype,
                                         i, MCA_COLL_BASE_TAG_ALLTOALLV,
                                         MCA_PML_BASE_SEND_STANDARD, comm,
                                         preq++));
            if (MPI_SUCCESS != err) { goto err_hndl; }
        }
    }

    /* Start your engines.  This will never return an error. */
    MCA_PML_CALL(start(nreqs, reqs));

    /* Wait for them all.  If there's an error, note that we don't care
     * what the error was -- just that there *was* an error.  The PML
     * will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error, and return the
     * error after we free everything. */
    err = ompi_request_wait_all(nreqs, reqs, MPI_STATUSES_IGNORE);

 err_hndl:
    /* find a real error code */
    if (MPI_ERR_IN_STATUS == err) {
        for( i = 0; i < nreqs; i++ ) {
            if (MPI_REQUEST_NULL == reqs[i]) continue;
            if (MPI_ERR_PENDING == reqs[i]->req_status.MPI_ERROR) continue;
            if (reqs[i]->req_status.MPI_ERROR != MPI_SUCCESS) {
                err = reqs[i]->req_status.MPI_ERROR;
                break;
            }
        }
    }
    /* Free the requests in all cases as they are persistent */
    ompi_coll_base_free_reqs(reqs, nreqs);

    return err;
}

/* Bruck pattern performs log2(commsize) iterations and use peers as
 * intermediary for data transfers. This reduces the number of communication but
 * increases the global amount of data transfered.
 * This algorithm targets small message sizes on large communicators.
 *
 * Ranks are implicitly renumered relatively to local rank
 * At iteration 1, data for peers 1, 11, 101, 111, 1001, ... are sent to rank +1
 * At iteration 2, data for peers 10, 11, 110, 111, 1010,... are sent to rank +2
 * At iteration 3, data for peers 100, 101, 110, 111, 1100 ... are sent to rank +4
 * ...
 * At the end data are arrived and can be placed in receive buffer.
 */
static void
alltoallv_bruck_update_blocks(const int *recv_block_size,
                              char *tmpbuf, int jump, int commsize,
                              int *block_count, ptrdiff_t *block_addr){
    int recv_offset = 0;
    int struct_idx = 0;
    for (int remote = 1; remote < commsize; ++remote){
        if( ! (remote & jump)) {
            continue;
        }
        /* update block statuses with recv blocks */
        block_count[remote] = recv_block_size[struct_idx];
        block_addr[remote] = (ptrdiff_t)tmpbuf + recv_offset;
        recv_offset += recv_block_size[struct_idx];
        ++ struct_idx;
    }
}

int
ompi_coll_base_alltoallv_intra_bruck(const void *sbuf, const int *scounts, const int *sdisps,
                                            struct ompi_datatype_t *sdtype,
                                            void *rbuf, const int *rcounts, const int *rdisps,
                                            struct ompi_datatype_t *rdtype,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    int line = -1;
    int rank;
    int commsize;
    int err = OMPI_SUCCESS;
    ptrdiff_t sext;
    ptrdiff_t rext;
    size_t ssize;

    int *sent_blocks_size = NULL;
    int *recv_block_size = NULL;
    int *block_used = NULL;
    int *block_count = NULL;
    ptrdiff_t *block_addr = NULL;
    int *struct_count = NULL;
    ptrdiff_t *struct_displs = NULL;
    ompi_datatype_t **struct_ddt = NULL;

    char **iter_tmpbuf = NULL;

    if (MPI_IN_PLACE == sbuf) {
        return mca_coll_base_alltoallv_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                            rdtype, comm, module);
    }

    commsize = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);


    err = ompi_datatype_type_extent (sdtype, &sext);
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
    err = ompi_datatype_type_size (sdtype, &ssize);
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
    err = ompi_datatype_type_extent (rdtype, &rext);
    if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

    /* Handle my block */
    ompi_datatype_sndrcv((const char*)sbuf + sdisps[rank] * sext, scounts[rank], sdtype,
                         (char*)rbuf + rdisps[rank] * rext, rcounts[rank], rdtype);

    if (commsize < 2) {
        return OMPI_SUCCESS;
    }
    /* Prepare iterations to send and receive other blocks */

    int n_iter = 0;
    for (int jump = 1; jump < commsize; jump <<=1) {
        ++n_iter;
    }
    /* Byte count for each remote, sorted in rotated order */
    sent_blocks_size = malloc(commsize * sizeof(int));
    recv_block_size = malloc(commsize * sizeof(int));

    block_used = malloc(commsize * sizeof(int));
    block_count = malloc(commsize * sizeof(int));
    block_addr = malloc(commsize * sizeof(ptrdiff_t));

    struct_count = malloc(commsize * sizeof(int));
    struct_displs = malloc(commsize * sizeof(int));
    struct_ddt = malloc(commsize * sizeof(ompi_datatype_t *));

    iter_tmpbuf = malloc(n_iter * sizeof(char*));

    if (NULL == sent_blocks_size || NULL == recv_block_size || NULL == block_used ||
        NULL == block_count || NULL == block_addr || NULL == struct_count ||
        NULL == struct_displs || NULL == struct_ddt || NULL == iter_tmpbuf){
        err = MPI_ERR_NO_SPACE; line = __LINE__; goto err_hndl;
    }

    for (int iter = 0; iter < n_iter; ++iter) {
        iter_tmpbuf[iter] = NULL;
    }

    /* local rotation */
    for (int i = 0; i < commsize; ++i) {
        int remote = (rank + i) % commsize;
        block_count[i] = scounts[remote];
        block_addr[i] = (ptrdiff_t)sbuf + sdisps[remote] * sext;
        block_used[i] = 0;
    }

    /* Communications iterations */

    for (int jump = 1, iter = 0; jump < commsize; jump <<=1, ++iter) {
        int sendto = (rank + jump) % commsize;
        int recvfrom = (rank - jump + commsize) % commsize;

        /* Prepare a derived datatype that contains data from this iteration. */
        /* For the receiver to allocate memory also prepare the description
         * of this datatype that will be sent just before the data */
        int n_to_send = 0;
        for (int remote = 1; remote < commsize; ++remote){
            if (! (remote & jump)) {
                /* this block do not move at this iteration */
                continue;
            }

            struct_count[n_to_send] = block_count[remote];
            struct_ddt[n_to_send] = MPI_BYTE;
            struct_displs[n_to_send] = block_addr[remote];

            sent_blocks_size[n_to_send] = block_count[remote];
            if (!block_used[remote]){
                block_used[remote] = 1;
                sent_blocks_size[n_to_send] *= ssize;
                struct_ddt[n_to_send] = sdtype;
            }

            ++ n_to_send;
        }

        ompi_datatype_t * send_struct;
        err = ompi_datatype_create_struct(n_to_send,
                                    struct_count,
                                    struct_displs,
                                    struct_ddt,
                                    &send_struct);
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
        err = ompi_datatype_commit(&send_struct);
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

        /* Perform actual communications; both sends and the recv of block sizes */
        int n_to_recv = n_to_send;
        ompi_request_t *reqs[4];

        err = MCA_PML_CALL(irecv(recv_block_size, n_to_recv, MPI_INT, recvfrom,
                       MCA_COLL_BASE_TAG_ALLTOALLV, comm, &reqs[0]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }


        err = MCA_PML_CALL(isend(sent_blocks_size, n_to_send, MPI_INT, sendto,
                       MCA_COLL_BASE_TAG_ALLTOALLV, MCA_PML_BASE_SEND_STANDARD,
                       comm, &reqs[1]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

        /* Do separate isend to let receiver to prepare room for the second message */
        err = MCA_PML_CALL(isend((char*)NULL, 1, send_struct, sendto,
                       MCA_COLL_BASE_TAG_ALLTOALLV, MCA_PML_BASE_SEND_STANDARD,
                       comm, &reqs[2]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }


        err = ompi_datatype_destroy(&send_struct);
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

        ompi_request_wait(&reqs[0], MPI_STATUS_IGNORE);
        /* Block sizes received: prepare memory to receive data */
        int total_recv = 0;
        for (int block_idx = 0; block_idx < n_to_recv; ++block_idx) {
            total_recv += recv_block_size[block_idx];
        }

        iter_tmpbuf[iter] = malloc(total_recv);
        err = MCA_PML_CALL(irecv(iter_tmpbuf[iter], total_recv, MPI_BYTE,
                                 recvfrom, MCA_COLL_BASE_TAG_ALLTOALLV,
                                 comm, &reqs[3]));
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
        alltoallv_bruck_update_blocks(recv_block_size,
                                      iter_tmpbuf[iter],
                                      jump,
                                      commsize,
                                      block_count,
                                      block_addr);
        err = ompi_request_wait_all(3, &reqs[1], MPI_STATUSES_IGNORE);
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }

    }

    /* End of communications iterations */

    /* Reverse rotation */
    /* TODO make communications to do reverse rotation */
    for (int remote = 0; remote < commsize; ++remote) {
        if (remote == rank) continue; /* Already done */
        int block_idx = (rank - remote + commsize) % commsize;
        err = ompi_datatype_sndrcv((char*)NULL + block_addr[block_idx],
                             block_count[block_idx], MPI_BYTE,
                             (char*)rbuf + rdisps[remote] * rext,
                             rcounts[remote], rdtype);
        if (err != MPI_SUCCESS) { line = __LINE__; goto err_hndl; }
    }

err_hndl:
    if (err != OMPI_SUCCESS) {
        opal_output(ompi_coll_base_framework.framework_output,
                    "ERROR %d found in %s at line %d", err, __func__, line);
    }

    if (NULL != iter_tmpbuf) {
        /* if an error occured early, iter_tmpbuf may be uninitiazed */
        for (int iter = 0; iter < n_iter; ++iter) {
            free(iter_tmpbuf[iter]);
        }
        free(iter_tmpbuf);
    }
    free(struct_ddt);
    free(struct_displs);
    free(struct_count);
    free(block_addr);
    free(block_count);
    free(block_used);
    free(recv_block_size);
    free(sent_blocks_size);
    return err;
}
