/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 */
#include "opal/runtime/opal_params.h"
#include "coll_bullnbc_internal.h"

static int nbc_allgatherv_intra_ring_init(const void* sendbuf, int sendcount,
                                MPI_Datatype sendtype, void* recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype,
                                struct ompi_communicator_t *comm,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module,
                                bool persistent);

/* The following are used by dynamic and forced rules */

static mca_base_var_enum_value_t allgatherv_algorithms[] = {
    {0, "legacy"},
    {1, "default"},
    {2, "bruck"},
    {3, "ring"},
    {4, "neighbor"},
    {5, "two_proc"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */
/* this routine is called by the component only */

int ompi_coll_bullnbc_allgatherv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != allgatherv_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iallgatherv_algorithm_count",
                                         "Number of allgatherv algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_allgatherv_algorithms", allgatherv_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iallgatherv_algorithm",
                                         "Which allgatherv algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "iallgatherv_algorithm_segmentsize",
                                  "Segment size in bytes used by default for iallgatherv algorithms. "
                                  "Only has meaning if algorithm is forced and supports segmenting. "
                                  "0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

/* an allgatherv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* simple linear MPI_Iallgatherv
 * the algorithm uses p-1 rounds
 * first round:
 *   each node sends to it's left node (rank+1)%p sendcount elements
 *   each node begins with it's right node (rank-11)%p and receives from it recvcounts[(rank+1)%p] elements
 * second round:
 *   each node sends to node (rank+2)%p sendcount elements
 *   each node receives from node (rank-2)%p recvcounts[(rank+2)%p] elements */
static int nbc_allgatherv_legacy_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                               void* recvbuf, const int *recvcounts, const int *displs,
                               MPI_Datatype recvtype, struct ompi_communicator_t *comm,
                               ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module,
                               bool persistent)
{
    MPI_Aint rcvext;
    char *rbuf, *sbuf, inplace;
    BULLNBC_Schedule *schedule;
    int rank, p, res;
    ompi_coll_bullnbc_module_t *bullnbc_module;


    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    NBC_IN_PLACE(sendbuf, recvbuf, inplace);

    rank = ompi_comm_rank (comm);
    p = ompi_comm_size (comm);

    res = ompi_datatype_type_extent (recvtype, &rcvext);
    if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
        NBC_Error ("MPI Error in ompi_datatype_type_extent() (%i)", res);
        return res;
    }
    if (inplace) {
        sendtype = recvtype;
        sendcount = recvcounts[rank];
    } else if (!persistent) { /* for persistent, the copy must be scheduled */
        /* copy my data to receive buffer */
        rbuf = (char *) recvbuf + displs[rank] * rcvext;
        res = NBC_Copy (sendbuf, sendcount, sendtype, rbuf, recvcounts[rank], recvtype, comm);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (NULL == schedule) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    sbuf = (char *) recvbuf + displs[rank] * rcvext;
    if (persistent && !inplace) { /* for nonblocking, data has been copied already */
        /* copy my data to receive buffer (= send buffer of NBC_Sched_send) */
        res = BULLNBC_Sched_copy ((void *)sendbuf, false, sendcount, sendtype,
                                  sbuf, false, recvcounts[rank], recvtype, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    }

    /* do p-1 rounds */
    for (int r = 1 ; r < p ; ++r) {
        int speer = (rank + r) % p;
        int rpeer = (rank - r + p) % p;
        rbuf = (char *)recvbuf + displs[rpeer] * rcvext;

        res = NBC_Sched_recv (rbuf, false, recvcounts[rpeer], recvtype, rpeer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }

        /* send to rank r - not from the sendbuf to optimize MPI_IN_PLACE */
        res = NBC_Sched_send (sbuf, false, recvcounts[rank], recvtype, speer, schedule, false);
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
    res = BULLNBC_Schedule_request (schedule, comm, bullnbc_module, persistent, request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
    }
    return OMPI_SUCCESS;
}

/*
 * ompi_coll_base_allgatherv_intra_neighbor
 *
 * Function:     allgatherv using N/2 steps (O(N))
 * Accepts:      Same arguments as MPI_Allgatherv
 * Returns:      MPI_SUCCESS or error code
 *
 * Description:  Neighbor Exchange algorithm for allgather adapted for
 *               allgatherv.
 *               Described by Chen et.al. in
 *               "Performance Evaluation of Allgather Algorithms on
 *                Terascale Linux Cluster with Fast Ethernet",
 *               Proceedings of the Eighth International Conference on
 *               High-Performance Computing inn Asia-Pacific Region
 *               (HPCASIA'05), 2005
 *
 *               Rank r exchanges message with one of its neighbors and
 *               forwards the data further in the next step.
 *
 *               No additional memory requirements.
 *
 * Limitations:  Algorithm works only on even number of processes.
 *               For odd number of processes we switch to ring algorithm.
 *
 * Example on 6 nodes:
 *  Initial state
 *    #     0      1      2      3      4      5
 *         [0]    [ ]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [1]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [3]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [4]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [ ]    [5]
 *   Step 0:
 *    #     0      1      2      3      4      5
 *         [0]    [0]    [ ]    [ ]    [ ]    [ ]
 *         [1]    [1]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [2]    [ ]    [ ]
 *         [ ]    [ ]    [3]    [3]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [4]    [4]
 *         [ ]    [ ]    [ ]    [ ]    [5]    [5]
 *   Step 1:
 *    #     0      1      2      3      4      5
 *         [0]    [0]    [0]    [ ]    [ ]    [0]
 *         [1]    [1]    [1]    [ ]    [ ]    [1]
 *         [ ]    [2]    [2]    [2]    [2]    [ ]
 *         [ ]    [3]    [3]    [3]    [3]    [ ]
 *         [4]    [ ]    [ ]    [4]    [4]    [4]
 *         [5]    [ ]    [ ]    [5]    [5]    [5]
 *   Step 2:
 *    #     0      1      2      3      4      5
 *         [0]    [0]    [0]    [0]    [0]    [0]
 *         [1]    [1]    [1]    [1]    [1]    [1]
 *         [2]    [2]    [2]    [2]    [2]    [2]
 *         [3]    [3]    [3]    [3]    [3]    [3]
 *         [4]    [4]    [4]    [4]    [4]    [4]
 *         [5]    [5]    [5]    [5]    [5]    [5]
 */
static int nbc_allgatherv_intra_neighbor_init(const void* sendbuf, int sendcount,
                                MPI_Datatype sendtype, void* recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype,
                                struct ompi_communicator_t *comm,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module,
                                bool persistent)
{
    BULLNBC_Schedule *schedule;
    int line, rank, size, i, even_rank, err = 0;
    int neighbor[2], offset_at_step[2], recv_data_from[2], send_data_from;
    int new_scounts[2], new_sdispls[2], new_rcounts[2], new_rdispls[2];
    ptrdiff_t slb, rlb, sext, rext;
    char *tmpsend = NULL, *tmprecv = NULL;
    struct ompi_datatype_t  *new_rdtype, *new_sdtype;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    if (size % 2) {
        return nbc_allgatherv_intra_ring_init(sendbuf, sendcount, sendtype, recvbuf,
                                              recvcounts, displs, recvtype, comm,
                                              request, module, persistent);
    }

    int nrounds = (size / 2);
    int arity[nrounds];

    for (i = 0; i < nrounds; i++) {
        arity[i] = 2;
    }

    schedule = BULLNBC_create_schedule(nrounds, arity, false);

    err = ompi_datatype_get_extent (sendtype, &slb, &sext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    err = ompi_datatype_get_extent (recvtype, &rlb, &rext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    /* Initialization step:
       - if send buffer is not MPI_IN_PLACE, copy send buffer to
       the appropriate block of receive buffer
    */
    tmprecv = (char*) recvbuf + (ptrdiff_t) displs[rank] * rext;
    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char*) sendbuf;
        err = ompi_datatype_sndrcv(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcounts[rank], recvtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
    }

    /* Determine neighbors, order in which blocks will arrive, etc. */
    even_rank = !(rank % 2);
    if (even_rank) {
        neighbor[0] = (rank + 1) % size;
        neighbor[1] = (rank - 1 + size) % size;
        recv_data_from[0] = rank;
        recv_data_from[1] = rank;
        offset_at_step[0] = (+2);
        offset_at_step[1] = (-2);
    } else {
        neighbor[0] = (rank - 1 + size) % size;
        neighbor[1] = (rank + 1) % size;
        recv_data_from[0] = neighbor[0];
        recv_data_from[1] = neighbor[0];
        offset_at_step[0] = (-2);
        offset_at_step[1] = (+2);
    }

    /* Communication loop:
       - First step is special: exchange a single block with neighbor[0].
       - Rest of the steps:
       update recv_data_from according to offset, and
       exchange two blocks with appropriate neighbor.
       the send location becomes previous receve location.
       Note, we need to create indexed datatype to send and receive these
       blocks properly.
    */
    tmprecv = (char*)recvbuf + (ptrdiff_t)displs[neighbor[0]] * rext;
    tmpsend = (char*)recvbuf + (ptrdiff_t)displs[rank] * rext;

    /* Sendreceive */
    err = BULLNBC_Sched_recv_insert (tmprecv, false, recvcounts[neighbor[0]],
                              recvtype, neighbor[0], schedule, 0, 0);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    err = BULLNBC_Sched_send_insert (tmpsend, false, recvcounts[rank], recvtype,
                              neighbor[0], schedule, 0, 1);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    /* Determine initial sending counts and displacements*/
    if (even_rank) {
        send_data_from = rank;
    } else {
        send_data_from = recv_data_from[0];
    }

    for (i = 1; i < (size / 2); i++) {
        const int i_parity = i % 2;
        recv_data_from[i_parity] =
            (recv_data_from[i_parity] + offset_at_step[i_parity] + size) % size;

        /* Create new indexed types for sending and receiving.
           We are sending data from ranks (send_data_from) and (send_data_from+1)
           We are receiving data from ranks (recv_data_from[i_parity]) and
           (recv_data_from[i_parity]+1).
        */
        new_scounts[0] = recvcounts[send_data_from];
        new_scounts[1] = recvcounts[(send_data_from + 1)];
        new_sdispls[0] = displs[send_data_from];
        new_sdispls[1] = displs[(send_data_from + 1)];
        err = ompi_datatype_create_indexed(2, new_scounts, new_sdispls, recvtype,
                                           &new_sdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        err = ompi_datatype_commit(&new_sdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

        new_rcounts[0] = recvcounts[recv_data_from[i_parity]];
        new_rcounts[1] = recvcounts[(recv_data_from[i_parity] + 1)];
        new_rdispls[0] = displs[recv_data_from[i_parity]];
        new_rdispls[1] = displs[(recv_data_from[i_parity] + 1)];
        err = ompi_datatype_create_indexed(2, new_rcounts, new_rdispls, recvtype,
                                           &new_rdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        err = ompi_datatype_commit(&new_rdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

        tmprecv = (char*) recvbuf;
        tmpsend = (char*) recvbuf;

        /* Sendreceive */
        err = BULLNBC_Sched_recv_insert (tmprecv, false, 1, new_rdtype, neighbor[i_parity],
                                   schedule, i, 0);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

        err = BULLNBC_Sched_send_insert (tmpsend, false, 1, new_sdtype, neighbor[i_parity],
                                   schedule, i, 1);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

        send_data_from = recv_data_from[i_parity];

        ompi_datatype_destroy(&new_sdtype);
        ompi_datatype_destroy(&new_rdtype);
    }

    err = BULLNBC_Sched_commit(schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    err = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent,
                        request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    return OMPI_SUCCESS;

 err_hndl:
    (void)line;  // silence compiler warning
    return err;
}

/*
 * ompi_coll_base_allgatherv_intra_bruck
 *
 * Function:     allgather using O(log(N)) steps.
 * Accepts:      Same arguments as MPI_Allgather
 * Returns:      MPI_SUCCESS or error code
 *
 * Description:  Variation to All-to-all algorithm described by Bruck et al.in
 *               "Efficient Algorithms for All-to-all Communications
 *                in Multiport Message-Passing Systems"
 * Note:         Unlike in case of allgather implementation, we relay on
 *               indexed datatype to select buffers appropriately.
 *               The only additional memory requirement is for creation of
 *               temporary datatypes.
 * Example on 7 nodes (memory lay out need not be in-order)
 *   Initial set up:
 *    #     0      1      2      3      4      5      6
 *         [0]    [ ]    [ ]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [1]    [ ]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [3]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [4]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [ ]    [5]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [ ]    [ ]    [6]
 *   Step 0: send message to (rank - 2^0), receive message from (rank + 2^0)
 *    #     0      1      2      3      4      5      6
 *         [0]    [ ]    [ ]    [ ]    [ ]    [ ]    [0]
 *         [1]    [1]    [ ]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [2]    [2]    [ ]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [3]    [3]    [ ]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [4]    [4]    [ ]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [5]    [5]    [ ]
 *         [ ]    [ ]    [ ]    [ ]    [ ]    [6]    [6]
 *   Step 1: send message to (rank - 2^1), receive message from (rank + 2^1).
 *           message contains all blocks from (rank) .. (rank + 2^2) with
 *           wrap around.
 *    #     0      1      2      3      4      5      6
 *         [0]    [ ]    [ ]    [ ]    [0]    [0]    [0]
 *         [1]    [1]    [ ]    [ ]    [ ]    [1]    [1]
 *         [2]    [2]    [2]    [ ]    [ ]    [ ]    [2]
 *         [3]    [3]    [3]    [3]    [ ]    [ ]    [ ]
 *         [ ]    [4]    [4]    [4]    [4]    [ ]    [ ]
 *         [ ]    [ ]    [5]    [5]    [5]    [5]    [ ]
 *         [ ]    [ ]    [ ]    [6]    [6]    [6]    [6]
 *   Step 2: send message to (rank - 2^2), receive message from (rank + 2^2).
 *           message size is "all remaining blocks"
 *    #     0      1      2      3      4      5      6
 *         [0]    [0]    [0]    [0]    [0]    [0]    [0]
 *         [1]    [1]    [1]    [1]    [1]    [1]    [1]
 *         [2]    [2]    [2]    [2]    [2]    [2]    [2]
 *         [3]    [3]    [3]    [3]    [3]    [3]    [3]
 *         [4]    [4]    [4]    [4]    [4]    [4]    [4]
 *         [5]    [5]    [5]    [5]    [5]    [5]    [5]
 *         [6]    [6]    [6]    [6]    [6]    [6]    [6]
 */
static int nbc_allgatherv_intra_bruck_init(const void* sendbuf, int sendcount,
                                MPI_Datatype sendtype, void* recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype,
                                struct ompi_communicator_t *comm,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module,
                                bool persistent)
{
    BULLNBC_Schedule *schedule;
    ompi_coll_bullnbc_module_t *bullnbc_module;
    int line, err = 0, rank, size, sendto, recvfrom, distance, i;
    int *new_rcounts = NULL, *new_rdispls = NULL, *new_scounts = NULL, *new_sdispls = NULL;
    ptrdiff_t slb, rlb, sext, rext;
    char *tmpsend = NULL, *tmprecv = NULL;
    struct ompi_datatype_t *new_rdtype, *new_sdtype;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* compute round number : dirty way */
    int nrounds = 0;

    for (distance = 1; distance < size; distance <<= 1) {
        nrounds +=1;
    }

    int arity[nrounds];

    for (i = 0; i < nrounds; i++) {
        arity[i] = 2;
    }

    schedule = BULLNBC_create_schedule(nrounds, arity, false);

    err = ompi_datatype_get_extent (sendtype, &slb, &sext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    err = ompi_datatype_get_extent (recvtype, &rlb, &rext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    /* Initialization step:
       - if send buffer is not MPI_IN_PLACE, copy send buffer to block rank of
       the receive buffer.
    */
    tmprecv = (char*) recvbuf + (ptrdiff_t) displs[rank] * rext;
    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char*) sendbuf;
        err = ompi_datatype_sndrcv(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcounts[rank], recvtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }

    }

    /* Communication step:
       At every step i, rank r:
       - doubles the distance
       - sends message with blockcount blocks, (rbuf[rank] .. rbuf[rank + 2^i])
       to rank (r - distance)
       - receives message of blockcount blocks,
       (rbuf[r + distance] ... rbuf[(r+distance) + 2^i]) from
       rank (r + distance)
       - blockcount doubles until the last step when only the remaining data is
       exchanged.
    */

    new_rcounts = (int*) calloc(4*size, sizeof(int));
    if (NULL == new_rcounts) { err = -1; line = __LINE__; goto err_hndl; }
    new_rdispls = new_rcounts + size;
    new_scounts = new_rdispls + size;
    new_sdispls = new_scounts + size;

    int roundid = 0;
    for (distance = 1; distance < size; distance<<=1) {
        int blockcount;
        recvfrom = (rank + distance) % size;
        sendto = (rank - distance + size) % size;

        if (distance <= (size >> 1)) {
            blockcount = distance;
        } else {
            blockcount = size - distance;
        }

        if(GPU_LIKELYHOOD(opal_iallgatherv_use_device_pointers)) {
            /* Buffers may be on GPU causing issues to some PML (UCX) to handle derived
             * datatypes. Send each block separately */
            for (i = 0; i < blockcount; i++) {
                const int tmp_srank = (rank + i) % size;
                const int tmp_rrank = (recvfrom + i) % size;
                err = BULLNBC_Sched_recv_insert (((char*)recvbuf)+ displs[tmp_rrank] * rext,
                                                 false, recvcounts[tmp_rrank], recvtype,
                                                 recvfrom, schedule, roundid, 0);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
                    OBJ_RELEASE(schedule);
                    return err;
                }

                err = BULLNBC_Sched_send_insert (((char*)recvbuf)+ displs[tmp_srank] * rext,
                                                 false, recvcounts[tmp_srank], recvtype,
                                                 sendto, schedule, roundid, 1);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
                    OBJ_RELEASE(schedule);
                    return err;
                }
            }
            roundid ++;
            continue;
        }

        /* create send and receive datatypes */
        for (i = 0; i < blockcount; i++) {
            const int tmp_srank = (rank + i) % size;
            const int tmp_rrank = (recvfrom + i) % size;
            new_scounts[i] = recvcounts[tmp_srank];
            new_sdispls[i] = displs[tmp_srank];
            new_rcounts[i] = recvcounts[tmp_rrank];
            new_rdispls[i] = displs[tmp_rrank];
        }

        err = ompi_datatype_create_indexed(blockcount, new_scounts, new_sdispls,
                                           recvtype, &new_sdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        err = ompi_datatype_create_indexed(blockcount, new_rcounts, new_rdispls,
                                           recvtype, &new_rdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

        err = ompi_datatype_commit(&new_sdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }
        err = ompi_datatype_commit(&new_rdtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

        /* Sendreceive */
        err = BULLNBC_Sched_recv_insert (recvbuf, false, 1, new_rdtype,
                                  recvfrom, schedule, roundid, 0);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

        err = BULLNBC_Sched_send_insert (recvbuf, false, 1, new_sdtype,
                                  sendto, schedule, roundid, 1);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

        roundid++;

        ompi_datatype_destroy(&new_sdtype);
        ompi_datatype_destroy(&new_rdtype);
    }

    free(new_rcounts);

    err = BULLNBC_Sched_commit(schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    err = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent,
                        request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    return OMPI_SUCCESS;

 err_hndl:
    if( NULL != new_rcounts ) free(new_rcounts);

    (void)line;  // silence compiler warning
    return err;
}

static int nbc_allgatherv_intra_ring_init(const void* sendbuf, int sendcount,
                                MPI_Datatype sendtype, void* recvbuf,
                                const int *recvcounts, const int *displs,
                                MPI_Datatype recvtype,
                                struct ompi_communicator_t *comm,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module,
                                bool persistent)
{
    BULLNBC_Schedule *schedule;
    int line, rank, size, sendto, recvfrom, i, recvdatafrom, senddatafrom, err;
    ptrdiff_t slb, rlb, sext, rext;
    char *tmpsend = NULL, *tmprecv = NULL;
    ompi_coll_bullnbc_module_t *bullnbc_module;

    bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    int arity[size];

    err = ompi_datatype_get_extent (sendtype, &slb, &sext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    err = ompi_datatype_get_extent (recvtype, &rlb, &rext);
    if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl; }

    /*
    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    */
    /* Initialization step:
       - if send buffer is not MPI_IN_PLACE, copy send buffer to
       the appropriate block of receive buffer
    */
    tmprecv = (char*) recvbuf + (ptrdiff_t) displs[rank] * rext;
    if (MPI_IN_PLACE != sendbuf) {
        tmpsend = (char*) sendbuf;
        err = ompi_datatype_sndrcv(tmpsend, sendcount, sendtype,
                                   tmprecv, recvcounts[rank],
                                   recvtype);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
    }

    /* Communication step:
       At every step i: 0 .. (P-1), rank r:
       - receives message from [(r - 1 + size) % size] containing data from rank
       [(r - i - 1 + size) % size]
       - sends message to rank [(r + 1) % size] containing data from rank
       [(r - i + size) % size]
       - sends message which starts at begining of rbuf and has size
    */
    sendto = (rank + 1) % size;
    recvfrom  = (rank - 1 + size) % size;


    for (i = 0; i < size; i++) {
        arity[i] = 2;
    }

    schedule = BULLNBC_create_schedule(size - 1, arity, false);

    for (i = 0; i < size - 1; i++) {
        recvdatafrom = (rank - i - 1 + size) % size;
        senddatafrom = (rank - i + size) % size;

        tmprecv = (char*)recvbuf + displs[recvdatafrom] * rext;
        tmpsend = (char*)recvbuf + displs[senddatafrom] * rext;

        err = BULLNBC_Sched_send_insert (tmpsend, false, recvcounts[senddatafrom], sendtype,
                    sendto, schedule, i, 0);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

        err = BULLNBC_Sched_recv_insert (tmprecv, false, recvcounts[recvdatafrom], recvtype,
                recvfrom, schedule, i, 1);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
            OBJ_RELEASE(schedule);
            return err;
        }

    }

    err = BULLNBC_Sched_commit(schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
      OBJ_RELEASE(schedule);
      return err;
    }

    err = BULLNBC_Schedule_request(schedule, comm, bullnbc_module, persistent,
                        request, NULL);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != err)) {
        OBJ_RELEASE(schedule);
        return err;
    }

    return OMPI_SUCCESS;

 err_hndl:
    (void)line;  // silence compiler warning
    return err;
}

int nbc_allgatherv_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                        const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                        struct ompi_communicator_t *comm, ompi_request_t ** request,
                        struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
    int algorithm = 0;

    if(mca_coll_bullnbc_component.use_dynamic_rules) {
        ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
        algorithm = mca_coll_bullnbc_component.forced_params[ALLGATHERV].algorithm;

        if(algorithm == 0 && bullnbc_module->com_rules[ALLGATHERV]) {
            size_t dsize;
            int i, comsize, total_size, per_rank_size, dummy1, dummy2, dummy3, res;


            res = ompi_datatype_type_size (recvtype, &dsize);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }

            total_size = 0;
            comsize = ompi_comm_size(comm);

            for (i = 0; i < comsize; i++) { total_size += dsize * recvcounts[i]; }

            per_rank_size = total_size / comsize;

            algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[ALLGATHERV],
                        per_rank_size, &dummy1, &dummy2, &dummy3);
        }
    }

    switch (algorithm) {
        case (0): return nbc_allgatherv_legacy_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                    displs, recvtype, comm, request, module, persistent);
        case (1): return nbc_allgatherv_legacy_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                    displs, recvtype, comm, request, module, persistent);
        case (2): return nbc_allgatherv_intra_bruck_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                         displs, recvtype, comm, request, module, persistent);
        case (3): return nbc_allgatherv_intra_ring_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                        displs, recvtype, comm, request, module, persistent);
        case (4): return nbc_allgatherv_intra_neighbor_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                            displs, recvtype, comm, request, module,persistent);
        case (5): return nbc_allgatherv_legacy_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts,
                                                    displs, recvtype, comm, request, module, persistent);
    }

    return OMPI_ERROR;
}


int ompi_coll_bullnbc_iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgatherv_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                                  comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    res = BULLNBC_Start(*(ompi_coll_bullnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        BULLNBC_Return_handle (*(ompi_coll_bullnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

static int nbc_allgatherv_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                     MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int res, rsize, arity, index;
  MPI_Aint rcvext;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
    NBC_Error ("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

    arity = 0;

    for (int r = 0 ; r < rsize ; ++r) {
        if (recvcounts[r]) {
            arity++;
        }
    }

    if (sendcount) {
        for (int r = 0 ; r < rsize ; ++r) {
            arity++;
        }
    }

  schedule = BULLNBC_create_schedule(1, &arity, false);
  if (NULL == schedule) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* do rsize  rounds */
  index = 0;
  for (int r = 0 ; r < rsize ; ++r) {
    char *rbuf = (char *) recvbuf + displs[r] * rcvext;

    if (recvcounts[r]) {
      res = BULLNBC_Sched_recv_insert (rbuf, false, recvcounts[r], recvtype, r, schedule, 0, index++);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  }

  if (sendcount) {
    for (int r = 0 ; r < rsize ; ++r) {
      res = BULLNBC_Sched_send_insert (sendbuf, false, sendcount, sendtype, r, schedule, 0, index++);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  }

  res = BULLNBC_Sched_commit (schedule);
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

int ompi_coll_bullnbc_iallgatherv_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                       MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgatherv_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                                        comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
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

int ompi_coll_bullnbc_allgatherv_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                     MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgatherv_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                                  comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_allgatherv_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                           MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                           struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgatherv_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype,
                                        comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
