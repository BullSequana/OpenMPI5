/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015-2017 Los Alamos National Security, LLC. All rights
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
#if OMPI_MPI_PARTITIONED_COLL
#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_partitioned_dag.h"
#include "opal/util/show_help.h"


static int palltoallvr_algorithm = 0;
#endif /* OMPI_MPI_PARTITIONED_COLL */

static inline int a2av_sched_linear(int rank, int p, BULLNBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts,
                                    const int *sdispls, MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts,
                                    const int *rdispls, MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_pairwise(int rank, int p, BULLNBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype);

static inline int a2av_sched_inplace(int rank, int p, BULLNBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap);

static mca_base_var_enum_value_t alltoallv_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "pairwise"},
    {0, NULL}
};

#if OMPI_MPI_PARTITIONED_COLL
static mca_base_var_enum_value_t palltoallvr_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "linear_per_partitions"},
    //{1, "partitioned_pt2pt"},
    {0, NULL}
};
#endif

/* The following are used by dynamic and forced rules */
/* this routine is called by the component only */

int ompi_coll_bullnbc_alltoallv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != alltoallv_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ialltoallv_algorithm_count",
                                         "Number of alltoallv algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_alltoallv_algorithms", alltoallv_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ialltoallv_algorithm",
                                         "Which alltoallv algorithm is used unless MPI_IN_PLACE flag has been specified. If any, a specific algorithm will be used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ialltoallv_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ialltoallv algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
#if OMPI_MPI_PARTITIONED_COLL
  palltoallvr_algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_palltoallvr_algorithms",
                                  palltoallvr_algorithms, &new_enum);
    int ret=
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "palltoallvr_algorithm",
                                         "Which palltoallvr algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0,
                                         MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &palltoallvr_algorithm);
  OBJ_RELEASE(new_enum);
#endif /* OMPI_MPI_PARTITIONED_COLL */
  return OMPI_SUCCESS;
}

/* an alltoallv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* simple linear Alltoallv */
static int nbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                              MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                              MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, p, res;
  MPI_Aint sndext=0, rcvext;
  BULLNBC_Schedule *schedule;
  char inplace;
  ptrdiff_t gap = 0;
  void * tmpbuf = NULL;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent (recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  enum { NBC_A2AV_LINEAR, NBC_A2AV_PAIRWISE, NBC_A2AV_INPLACE } alg = NBC_A2AV_LINEAR;
  if(inplace) {
    alg = NBC_A2AV_INPLACE;
  } else if(mca_coll_bullnbc_component.use_dynamic_rules) {
    if(0 != mca_coll_bullnbc_component.forced_params[ALLTOALLV].algorithm) {
      alg = mca_coll_bullnbc_component.forced_params[ALLTOALLV].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
    } else if(bullnbc_module->com_rules[ALLTOALLV]) {
      int algorithm,dummy1,dummy2,dummy3;
      /**
       * check to see if we have some filebased rules. As we don't have global
       * knowledge about the total amount of data, use the first available rule.
       * This allow the users to specify the alltoallv algorithm to be used only
       * based on the communicator size.
      */
      algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[ALLTOALLV],
                                                           0, &dummy1, &dummy2, &dummy3);
      if(algorithm) {
        alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
      }
    }
  }

  opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                      "Bullnbc ialltoallv : algorithm %d (no segmentation supported)",
                      alg + 1);
  /* copy data to receivbuffer */
  if (inplace) {
    int count = 0;
    for (int i = 0; i < p; i++) {
      if (recvcounts[i] > count) {
        count = recvcounts[i];
      }
    }
    ptrdiff_t span = opal_datatype_span(&recvtype->super, count, &gap);
    if (OPAL_UNLIKELY(0 == span)) {
      return bullnbc_get_noop_request(persistent, request);
    }
    tmpbuf = malloc(span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
    sendcounts = recvcounts;
    sdispls = rdispls;
  } else {
    res = ompi_datatype_type_extent (sendtype, &sndext);
    if (MPI_SUCCESS != res) {
      NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
      return res;
    }
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }


  if (!inplace && sendcounts[rank] != 0) {
    char *rbuf = (char *) recvbuf + rdispls[rank] * rcvext;
    char *sbuf = (char *) sendbuf + sdispls[rank] * sndext;
    res = NBC_Sched_copy (sbuf, false, sendcounts[rank], sendtype,
                          rbuf, false, recvcounts[rank], recvtype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }
  }

  switch(alg) {
    case NBC_A2AV_INPLACE:
      res = a2av_sched_inplace(rank, p, schedule, recvbuf, recvcounts,
                               rdispls, rcvext, recvtype, gap);
      break;
    case NBC_A2AV_LINEAR:
      res = a2av_sched_linear(rank, p, schedule,
                              sendbuf, sendcounts, sdispls, sndext, sendtype,
                              recvbuf, recvcounts, rdispls, rcvext, recvtype);
      break;
    case NBC_A2AV_PAIRWISE:
      res = a2av_sched_pairwise(rank, p, schedule,
                                sendbuf, sendcounts, sdispls, sndext, sendtype,
                                recvbuf, recvcounts, rdispls, rcvext, recvtype);
      break;
  }

  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
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

int ompi_coll_bullnbc_ialltoallv(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype,
                                 recvbuf, recvcounts, rdispls, recvtype,
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

/* simple linear Alltoallv */
static int nbc_alltoallv_inter_init (const void* sendbuf, const int *sendcounts, const int *sdispls,
                                     MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                     MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int res, rsize;
  MPI_Aint sndext, rcvext;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;


  res = ompi_datatype_type_extent(sendtype, &sndext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent(recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  rsize = ompi_comm_remote_size (comm);

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int i = 0; i < rsize; i++) {
    /* post all sends */
    if (sendcounts[i] != 0) {
      char *sbuf = (char *) sendbuf + sdispls[i] * sndext;
      res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
    /* post all receives */
    if (recvcounts[i] != 0) {
      char *rbuf = (char *) recvbuf + rdispls[i] * rcvext;
      res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
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

int ompi_coll_bullnbc_ialltoallv_inter (const void* sendbuf, const int *sendcounts, const int *sdispls,
				       MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
				       MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				       struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype,
                                       recvbuf, recvcounts, rdispls, recvtype,
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

__opal_attribute_unused__
static inline int a2av_sched_linear(int rank, int p, BULLNBC_Schedule *schedule,
                                    const void *sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Aint sndext, MPI_Datatype sendtype,
                                    void *recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Aint rcvext, MPI_Datatype recvtype) {
  int res;

  for (int i = 0 ; i < p ; ++i) {
    if (i == rank) {
      continue;
    }

    /* post send */
    if (sendcounts[i] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[i] * sndext);
      res = NBC_Sched_send(sbuf, false, sendcounts[i], sendtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[i] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[i] * rcvext);
      res = NBC_Sched_recv(rbuf, false, recvcounts[i], recvtype, i, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

__opal_attribute_unused__
static inline int a2av_sched_pairwise(int rank, int p, BULLNBC_Schedule *schedule,
                                      const void *sendbuf, const int *sendcounts, const int *sdispls,
                                      MPI_Aint sndext, MPI_Datatype sendtype,
                                      void *recvbuf, const int *recvcounts, const int *rdispls,
                                      MPI_Aint rcvext, MPI_Datatype recvtype) {
  int res;

  for (int i = 1 ; i < p ; ++i) {
    int sndpeer = (rank + i) % p;
    int rcvpeer = (rank + p - i) %p;

    /* post send */
    if (sendcounts[sndpeer] != 0) {
      char *sbuf = ((char *) sendbuf) + (sdispls[sndpeer] * sndext);
      res = NBC_Sched_send(sbuf, false, sendcounts[sndpeer], sendtype, sndpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* post receive */
    if (recvcounts[rcvpeer] != 0) {
      char *rbuf = ((char *) recvbuf) + (rdispls[rcvpeer] * rcvext);
      res = NBC_Sched_recv(rbuf, false, recvcounts[rcvpeer], recvtype, rcvpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2av_sched_inplace(int rank, int p, BULLNBC_Schedule *schedule,
                                    void *buf, const int *counts, const int *displs,
                                    MPI_Aint ext, MPI_Datatype type, ptrdiff_t gap) {
  int res;

  for (int i = 1; i < (p+1)/2; i++) {
    int speer = (rank + i) % p;
    int rpeer = (rank + p - i) % p;
    char *sbuf = (char *) buf + displs[speer] * ext;
    char *rbuf = (char *) buf + displs[rpeer] * ext;

    if (0 != counts[rpeer]) {
      res = NBC_Sched_copy (rbuf, false, counts[rpeer], type,
                            (void *)(-gap), true, counts[rpeer], type,
                            schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
      res = NBC_Sched_send (sbuf, false , counts[speer], type, speer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[rpeer]) {
      res = NBC_Sched_recv (rbuf, false , counts[rpeer], type, rpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
      res = NBC_Sched_send ((void *)(-gap), true, counts[rpeer], type, rpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
    if (0 != counts[speer]) {
      res = NBC_Sched_recv (sbuf, false, counts[speer], type, speer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }
  if (0 == (p%2)) {
    int peer = (rank + p/2) % p;

    char *tbuf = (char *) buf + displs[peer] * ext;
    res = NBC_Sched_copy (tbuf, false, counts[peer], type,
                          (void *)(-gap), true, counts[peer], type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    if (0 != counts[peer]) {
      res = NBC_Sched_send ((void *)(-gap), true , counts[peer], type, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
      res = NBC_Sched_recv (tbuf, false , counts[peer], type, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoallv_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                 comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_alltoallv_inter_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                          MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoallv_inter_init(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype,
                                       comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

#if OMPI_MPI_PARTITIONED_COLL
static int sum(const int *val, size_t n)
{
    int ret = 0;
    for (size_t i = 0; i < n; ++i) {
        ret += val[i];
    }
    return ret;
}
static int max(const int *val, size_t n)
{
    if (n <= 0) {
        NBC_Error("Bad array size");
    }
    int ret = val[0]; /* n-1 checks that n>0 */
    for (size_t i = 1; i < n; ++i) {
        if ( ret < val[i]) {
            ret = val[i];
        }
    }
    return ret;
}

/*  Schedule a partitioned send/recv for each peer. This is a linear pattern:
 *  all data are directly sent to their final target.
 *  However partitions corresponding to the same target are aggregated */
static int
palltoallvr_init_linear(const void *sbuf,
                        const int *sparts, const int *scounts,
                        const int *sdispls,
                        struct ompi_datatype_t *sdtype,
                        ompi_request_t ** sendreqs,
                        void* rbuf,
                        const int *rparts, const int *rcounts,
                        const int *rdispls,
                        struct ompi_datatype_t *rdtype,
                        ompi_request_t ** recvreqs,
                        struct ompi_communicator_t *comm,
                        struct ompi_info_t *info,
                        ompi_request_t ** request,
                        struct mca_coll_base_module_2_4_0_t *module)
{
    MPI_Aint rcvext, sndext;
    int ret = OMPI_SUCCESS, comm_size;
    comm_size = ompi_comm_size (comm);

    ret = ompi_datatype_type_extent(sdtype, &sndext);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }
    ret = ompi_datatype_type_extent(rdtype, &rcvext);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }

    /* Compute collective caracteristics to alloc a request */
    int tot_sparts = sum(sparts, comm_size);
    int tot_rparts = sum(rparts, comm_size);
    int nsend = comm_size;
    int nrecv = comm_size;
    int n_intern_reqs = nsend + nrecv;
    int n_immediate_start = nrecv;
    for (int peer=0; peer < comm_size; ++peer){
        if (0 == sparts[peer] || 0 == scounts[peer] || 0 == sndext) {
            n_intern_reqs --;
        }
        if (0 == rparts[peer] || 0 == rcounts[peer] || 0 == rcvext) {
            /* cancel recv but prepare completion on start */
            n_immediate_start += -1 + rparts[peer];
            n_intern_reqs --;
        }
    }

    /* Collective request allocation */
    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(tot_sparts, sendreqs,
                                                   tot_rparts, recvreqs,
                                                   n_intern_reqs,
                                                   n_immediate_start,
                                                   comm,  request);
    int part_offset = 0;

    /* Schedule a single send to each remote, triggered for each peer when
     * all its partitions get ready.
     * Send completion triggers send partitions completion */
    for (int peer=0; peer < comm_size; ++peer){
        const char * peer_sbuf = ((char*)sbuf) + sdispls[peer] * sndext;
        int peer_count = sparts[peer] * scounts[peer];

        pcoll_dag_item * send = NULL;
        if (peer_count  && sndext) {
            send =  sched_send(peer_sbuf,
                               peer_count,
                               sdtype,
                               peer,
                               comm, NULL, req,
                               sparts[peer]);
        }

        for (int part=0; part < sparts[peer]; ++part){
            int part_idx = part_offset + part;
            pcoll_dag_item * ready = sched_ready(part_idx, req, 1);
            pcoll_dag_item * scomplete = sched_complete (part_idx, req);
            if (NULL != send) {
                create_dependency(ready, send);
                create_dependency(send, scomplete);
            } else {
                NBC_DEBUG(5, "Skip send of %dx%d to peer %d\n",
                          peer, sparts[peer], scounts[peer]);
                create_dependency(ready, scomplete);
            }
        }

        part_offset += sparts[peer];
    }

    /* Schedule a single recv to each remote.
     * Recv completion triggers recv partition completion */
    for (int peer=0; peer < comm_size; ++peer){
        char * peer_rbuf = ((char*)rbuf) + rdispls[peer] * rcvext;
        int peer_count = rparts[peer] * rcounts[peer];
        pcoll_dag_item * recv = NULL;
        if (peer_count  && rcvext) {
            recv =  sched_recv_v2(peer_rbuf,
                                  peer_count,
                                  rdtype,
                                  peer,
                                  comm, req);
        }
        for (int part=0; part < rparts[peer]; ++part){
            int part_idx = part_offset + part;
            pcoll_dag_item * rcomplete = sched_complete (part_idx, req);
            if (NULL != recv) {
                create_dependency(recv, rcomplete);
            } else {
                NBC_DEBUG(5, "Skip recv from peer %d, data count = %dx%d\n",
                          peer , rparts[peer], rcounts[peer]);
                run_on_start(rcomplete, req);
            }
        }

        part_offset += rparts[peer];
    }

    *request = &req->req_ompi;
    return OMPI_SUCCESS;
}

/*  Schedule a partitioned send/recv for each peer. This is a linear pattern:
 *  all data are directly sent to their final target.
 *  No aggregations of partitions */
static int
palltoallvr_init_linear_per_parts(const void *sbuf,
                                  const int *sparts, const int *scounts,
                                  const int *sdispls,
                                  struct ompi_datatype_t *sdtype,
                                  ompi_request_t ** sendreqs,
                                  void* rbuf,
                                  const int *rparts, const int *rcounts,
                                  const int *rdispls,
                                  struct ompi_datatype_t *rdtype,
                                  ompi_request_t ** recvreqs,
                                  struct ompi_communicator_t *comm,
                                  struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module)
{
    MPI_Aint rcvext, sndext;
    int ret = OMPI_SUCCESS, comm_size;
    comm_size = ompi_comm_size (comm);


    int tot_sparts = sum( sparts, comm_size);
    int tot_rparts = sum( rparts, comm_size);
    int nsend = tot_sparts;
    int nrecv = tot_rparts;
    int max_spart = max(sparts, comm_size);
    int max_rpart = max(rparts, comm_size);
    int max_part = max_spart> max_rpart ? max_spart : max_rpart;
    int max_tag;
    /* Iallreduce ? */
    comm->c_coll->coll_allreduce(&max_part, &max_tag, 1, MPI_INT, MPI_MAX, comm, comm->c_coll->coll_allreduce_module);

    ret = ompi_datatype_type_extent(sdtype, &sndext);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }
    ret = ompi_datatype_type_extent(rdtype, &rcvext);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }

    /* Compute collective caracteristics to alloc a request */
    int n_intern_reqs = nsend + nrecv;
    int n_immediate_start = nrecv;
    for (int peer=0; peer < comm_size; ++peer){
        if (0 == sparts[peer] || 0 == scounts[peer] || 0 == sndext) {
            n_intern_reqs -= sparts[peer];
        }
        if (0 == rparts[peer] || 0 == rcounts[peer] || 0 == rcvext) {
            n_intern_reqs -= rparts[peer];
            /* cancel recv but prepare completion on start */
        }
    }


    /* Collective request allocation */
    ompi_coll_bullnbc_pcoll_request_t * req;
    req = coll_bullnbc_pcoll_init_dag_ntag_req(tot_sparts, sendreqs,
                                                   tot_rparts, recvreqs,
                                                   n_intern_reqs, n_immediate_start,
                                                   comm, max_tag, request);
    int part_offset = 0;

    /* Schedule individual send for each partition, triggered when
     * partition gets ready. Send completion triggers partitions completion */
    for (int peer=0; peer < comm_size; ++peer){
        const char * peer_sbuf = ((char*)sbuf) + sdispls[peer] * sndext;

        for (int part=0; part < sparts[peer]; ++part){
            int part_idx = part_offset + part;
            pcoll_dag_item * ready = sched_ready(part_idx, req, 1);
            pcoll_dag_item * scomplete = sched_complete (part_idx, req);

            if (sparts[peer] && scounts[peer] && sndext) {
                const char * part_buf = peer_sbuf + part * sndext *scounts[peer];
                pcoll_dag_item * send =  sched_send_tagged(part_buf,
                                                           scounts[peer],
                                                           sdtype,
                                                           peer, part,
                                                           comm, NULL, req,
                                                           sparts[peer]);

                create_dependency(ready, send);
                create_dependency(send, scomplete);
            } else {
                NBC_DEBUG(5, "Skip send of %dx%d to peer %d\n",
                          peer, sparts[peer], scounts[peer]);
                create_dependency(ready, scomplete);
            }
        }

        part_offset += sparts[peer];
    }

    /* Schedule individual recv for each partition.
     * Recv completion triggers partitions completion */
    for (int peer=0; peer < comm_size; ++peer){
        char * peer_rbuf = ((char*)rbuf) + rdispls[peer] * rcvext;
        for (int part=0; part < rparts[peer]; ++part){
            int part_idx = part_offset + part;
            pcoll_dag_item * rcomplete = sched_complete (part_idx, req);

            if (rparts[peer] && rcounts[peer] && rcvext) {
                char * part_buf = peer_rbuf + part * rcvext * rcounts[peer];
                pcoll_dag_item * recv =  sched_recv_tagged_v2(part_buf,
                                                              rcounts[peer],
                                                              rdtype,
                                                              peer, part,
                                                              comm, req);

                create_dependency(recv, rcomplete);
            } else {
                NBC_DEBUG(5, "Skip recv from peer %d, data count = %dx%d\n",
                          peer , rparts[peer], rcounts[peer]);
                run_on_start(rcomplete, req);
            }
        }

        part_offset += rparts[peer];
    }

    *request = &req->req_ompi;
    return OMPI_SUCCESS;
}

static int total_count(const int *parts, const int * counts, size_t n) {
    /* TODO use a dot product, with sve support */
    int ret = 0;
    for (size_t i = 0; i < n; ++i) {
        ret += parts[i] * counts[i];
    }
    return ret;
}
int
ompi_coll_bullnbc_palltoallvr_init(const void *sbuf,
                                   const int *sparts, const int *scounts,
                                   const int *sdispls,
                                   struct ompi_datatype_t *sdtype,
                                   ompi_request_t ** sendreqs,
                                   void* rbuf,
                                   const int *rparts, const int *rcounts,
                                   const int *rdispls,
                                   struct ompi_datatype_t *rdtype,
                                   ompi_request_t ** recvreqs,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module)
{
    char inplace;
    size_t rdtype_size, total_rsize;
    size_t sdtype_size, total_ssize;
    unsigned int comm_size, total_sparts, total_rparts;
    comm_size = ompi_comm_size(comm);
    ompi_datatype_type_size(rdtype, &rdtype_size);
    ompi_datatype_type_size(sdtype, &sdtype_size);

    /* MPIX internal interface garantees that sparts, scounts and sdispls are valid */
    total_ssize = total_count(sparts, scounts, comm_size) * sdtype_size;
    total_rsize = total_count(rparts, rcounts, comm_size) * rdtype_size;
    total_sparts = sum(sparts, comm_size);
    total_rparts = sum(rparts, comm_size);

    NBC_IN_PLACE(sbuf, rbuf, inplace);
    if (inplace && total_rsize) {
        /* TODO fallback */
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "No MPI_IN_PLACE for partitioned", true,
                       "palltoallr");
        ompi_mpi_abort(comm, 0);
    }

    int alg = palltoallvr_algorithm;
    if (0 == alg) {
            alg = 1;
    }

    static int run=0;
    NBC_DEBUG(20, "Starting alltoallv %d, alg%d\n", run ++, alg);

    if (! mca_coll_bullnbc_uniform_partition_sizes && 2 == alg){
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Non uniform partition sizes", true,
                       2, "alltoallv", 1);
        alg = 1;
    }

    if (0 == total_ssize && 0 == total_rsize &&
        /* Alg linear per part needs an allreduce on part counts,
         * This may be the only process with no data to transfer.
         * It needs to assist the allreduce */
        alg != 2) {
        /* TODO do not perform 0 length sends or recvs */
        ompi_mca_coll_bullnbc_pcoll_init_empty_req(total_sparts, sendreqs,
                                                   total_rparts, recvreqs,
                                                   comm, request);
        return MPI_SUCCESS;
    }

    module_need_progress((ompi_coll_bullnbc_module_t*) module);

    switch(alg){
        case 1:
            return palltoallvr_init_linear(sbuf, sparts, scounts, sdispls,
                                           sdtype, sendreqs,
                                           rbuf, rparts, rcounts, rdispls,
                                           rdtype, recvreqs,
                                           comm, info, request, module);
        case 2:
            return palltoallvr_init_linear_per_parts(sbuf, sparts, scounts, sdispls,
                                                     sdtype, sendreqs,
                                                     rbuf, rparts, rcounts, rdispls,
                                                     rdtype, recvreqs,
                                                     comm, info, request, module);
        default:
            fprintf(stderr, "Palltoallr algorithm %d is not implemented yet\n", alg); fflush(stderr);
            abort();
    }
}

int
ompi_coll_bullnbc_palltoallv_init(const void *sbuf,
                                  const int *sparts, const int *scounts,
                                  const int *sdispls,
                                  struct ompi_datatype_t *sdtype,
                                  void* rbuf,
                                  const int *rparts, const int *rcounts,
                                  const int *rdispls,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module)
{
    return ompi_coll_bullnbc_palltoallvr_init(sbuf, sparts,scounts, sdispls, sdtype,
                                              MPIX_NO_REQUESTS,
                                              rbuf, rparts, rcounts, rdispls, rdtype,
                                              MPIX_NO_REQUESTS,
                                              comm, info, request, module);
}
#endif /* OMPI_MPI_PARTITIONED_COLL*/
