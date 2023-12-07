/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2017 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC.  All rights
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
#include "coll_bullnbc_internal.h"

static inline int allgather_sched_linear(
    int rank, int comm_size, BULLNBC_Schedule *schedule, const void *sendbuf,
    int scount, struct ompi_datatype_t *sdtype, void *recvbuf, int rcount,
    struct ompi_datatype_t *rdtype);
static inline int allgather_sched_recursivedoubling(
    int rank, int comm_size, BULLNBC_Schedule *schedule, const void *sbuf,
    int scount, struct ompi_datatype_t *sdtype, void *rbuf, int rcount,
    struct ompi_datatype_t *rdtype);

static mca_base_var_enum_value_t allgather_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "recursive_doubling"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */
/* this routine is called by the component only */

int ompi_coll_bullnbc_allgather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != allgather_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iallgather_algorithm_count",
                                         "Number of allgather algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_allgather_algorithms", allgather_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iallgather_algorithm",
                                         "Which allgather algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "iallgather_algorithm_segmentsize",
                                  "Segment size in bytes used by default for iallgather algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

static int nbc_allgather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                              MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, p, res;
  size_t size;
  MPI_Aint rcvext;
  BULLNBC_Schedule *schedule;
  char *rbuf, inplace;
  int is_commsize_pow2;
  enum { NBC_ALLGATHER_LINEAR, NBC_ALLGATHER_RDBL} alg;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    return res;
  }

  if (inplace) {
    sendtype = recvtype;
    sendcount = recvcount;
  } else if (!persistent) { /* for persistent, the copy must be scheduled */
    /* copy my data to receive buffer */
    rbuf = (char *) recvbuf + rank * recvcount * rcvext;
    res = NBC_Copy (sendbuf, sendcount, sendtype, rbuf, recvcount, recvtype, comm);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }
  if (1 == p && (!persistent || inplace)) {
    return bullnbc_get_noop_request(persistent, request);
  }

    res = ompi_datatype_type_size (recvtype, &size);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    size *= recvcount;

    if(mca_coll_bullnbc_component.use_dynamic_rules) {
        if(0 != mca_coll_bullnbc_component.forced_params[ALLGATHER].algorithm) {
            /* if op is not commutative or MPI_IN_PLACE was specified we have to deal with it */
            alg = mca_coll_bullnbc_component.forced_params[ALLGATHER].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
            goto selected_rule;
        }
        if(bullnbc_module->com_rules[ALLGATHER]) {
            int algorithm,dummy1,dummy2,dummy3;
            algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[ALLGATHER],
                                                                 size, &dummy1, &dummy2, &dummy3);
            if(algorithm) {
                alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
                goto selected_rule;
            }
        }
    }

    alg = NBC_ALLGATHER_LINEAR;
selected_rule:

    is_commsize_pow2 = !(p & (p - 1));
    if(alg == NBC_ALLGATHER_RDBL && !is_commsize_pow2){
        if(rank == 0){
            NBC_DEBUG(1,"Comm size prevents the usage of the recursivedoubling iallgather\n");
        }
        alg= NBC_ALLGATHER_LINEAR;
    }


    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (persistent && !inplace) {
      /* for nonblocking, data has been copied already */
      /* copy my data to receive buffer (= send buffer of NBC_Sched_send) */
      rbuf = (char *)recvbuf + rank * recvcount * rcvext;
      res = NBC_Sched_copy((void *)sendbuf, false, sendcount, sendtype,
                            rbuf, false, recvcount, recvtype, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }

    switch (alg) {
      case NBC_ALLGATHER_LINEAR:
        res = allgather_sched_linear(rank, p, schedule, sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype);
        break;
      case NBC_ALLGATHER_RDBL:
        res = allgather_sched_recursivedoubling(rank, p, schedule, sendbuf, sendcount,
                                                sendtype, recvbuf, recvcount, recvtype);
        break;
    }
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    res = NBC_Sched_commit(schedule);
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

int ompi_coll_bullnbc_iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module)
{
    int res = nbc_allgather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
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

static int nbc_allgather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int res, rsize;
  MPI_Aint rcvext;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error ("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  rsize = ompi_comm_remote_size (comm);

  /* set up schedule */
  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* do rsize - 1 rounds */
  for (int r = 0 ; r < rsize ; ++r) {
    /* recv from rank r */
    char *rbuf = (char *) recvbuf + r * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, r, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    /* send to rank r */
    res = NBC_Sched_send (sendbuf, false, sendcount, sendtype, r, schedule, false);
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

int ompi_coll_bullnbc_iallgather_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
				      MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				      struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
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

/*
 * allgather_sched_linear
 *
 * Description: an implementation of Iallgather using linear algorithm
 *
 * Time: O(comm_size)
 * Schedule length (rounds): O(comm_size)
 */
static inline int allgather_sched_linear(
    int rank, int comm_size, BULLNBC_Schedule *schedule, const void *sendbuf,
    int scount, struct ompi_datatype_t *sdtype, void *recvbuf, int rcount,
    struct ompi_datatype_t *rdtype)
{
    int res;
    ptrdiff_t rlb, rext;

    res = ompi_datatype_get_extent(rdtype, &rlb, &rext);
    char *sbuf = (char *)recvbuf + rank * rcount * rext;

    for (int remote = 0; remote < comm_size ; ++remote) {
        if (remote != rank) {
            /* Recv from rank remote */
            char *rbuf = (char *)recvbuf + remote * rcount * rext;
            res = NBC_Sched_recv(rbuf, false, rcount, rdtype, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Send to rank remote - not from the sendbuf to optimize MPI_IN_PLACE */
            res = NBC_Sched_send(sbuf, false, rcount, rdtype, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
        }
    }

cleanup_and_return:
    return res;
}

/*
 * allgather_sched_recursivedoubling
 *
 * Description: an implementation of Iallgather using recursive doubling algorithm
 * Limitation: power-of-two number of processes only
 * Time: O(log(comm_size))
 * Schedule length (rounds): O(log(comm_size))
 * Memory: no additional memory requirements beyond user-supplied buffers.
 *
 * Example on 4 nodes:
 *   Initialization: everyone has its own buffer at location rank in rbuf
 *    #     0      1      2      3
 *         [0]    [ ]    [ ]    [ ]
 *         [ ]    [1]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [ ]
 *         [ ]    [ ]    [ ]    [3]
 *   Step 0: exchange data with (rank ^ 2^0)
 *    #     0      1      2      3
 *         [0]    [0]    [ ]    [ ]
 *         [1]    [1]    [ ]    [ ]
 *         [ ]    [ ]    [2]    [2]
 *         [ ]    [ ]    [3]    [3]
 *   Step 1: exchange data with (rank ^ 2^1) (if you can)
 *    #     0      1      2      3
 *         [0]    [0]    [0]    [0]
 *         [1]    [1]    [1]    [1]
 *         [2]    [2]    [2]    [2]
 *         [3]    [3]    [3]    [3]
 *
 */
static inline int allgather_sched_recursivedoubling(
    int rank, int comm_size, BULLNBC_Schedule *schedule, const void *sbuf,
    int scount, struct ompi_datatype_t *sdtype, void *rbuf, int rcount,
    struct ompi_datatype_t *rdtype)
{
    int res;
    ptrdiff_t rlb, rext;
    char *tmpsend = NULL, *tmprecv = NULL;

    res = ompi_datatype_get_extent(rdtype, &rlb, &rext);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

    int sendblocklocation = rank;
    for (int distance = 1; distance < comm_size; distance <<= 1) {
        int remote = rank ^ distance;

        tmpsend = (char *)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
        if (rank < remote) {
            tmprecv = (char *)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
        } else {
            tmprecv = (char *)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
            sendblocklocation -= distance;
        }

        res = NBC_Sched_send(tmpsend, false, (ptrdiff_t)distance * (ptrdiff_t)rcount,
                             rdtype, remote, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

        res = NBC_Sched_recv(tmprecv, false, (ptrdiff_t)distance * (ptrdiff_t)rcount,
                             rdtype, remote, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
    }

cleanup_and_return:
    return res;
}

int ompi_coll_bullnbc_allgather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                 comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_allgather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_allgather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                       comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
