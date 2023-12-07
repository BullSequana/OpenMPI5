/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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
#if OMPI_MPI_NOTIFICATIONS
#include "ompi/mpiext/notified_rma/c/mpiext_notifs_rma.h"
#endif

#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_partitioned_dag.h"
#include "opal/util/show_help.h"

static inline int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule *schedule,
                                   const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static inline int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule *schedule,
                                     const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                     int recvcount, MPI_Datatype recvtype, MPI_Comm comm);
static inline int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule* schedule,
                                 const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                 int recvcount, MPI_Datatype recvtype, MPI_Comm comm, void* tmpbuf);
static inline int a2a_sched_inplace(int rank, int p, BULLNBC_Schedule* schedule, void* buf, int count,
                                   MPI_Datatype type, MPI_Aint ext, ptrdiff_t gap, MPI_Comm comm);

static int
ompi_coll_bullnbc_palltoallr_part_pt2pt (const void *sbuf, int sparts,
                                         int scount, struct ompi_datatype_t *sdtype,
                                         ompi_request_t ** srequest,
                                         void* rbuf, int rparts,
                                         int rcount, struct ompi_datatype_t *rdtype,
                                         ompi_request_t ** rrequest,
                                         struct ompi_communicator_t *comm,
                                         struct ompi_info_t *info,
                                         ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);
static int
ompi_coll_bullnbc_palltoallr_init_bruck (const void *sbuf, int sparts,
                                         int scount, ompi_datatype_t *sdtype,
                                         ompi_request_t ** srequest,
                                         void* rbuf, int rparts,
                                         int rcount, ompi_datatype_t *rdtype,
                                         ompi_request_t ** rrequest,
                                         struct ompi_communicator_t *comm,
                                         struct ompi_info_t *info,
                                         ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);
static int
ompi_coll_bullnbc_palltoallr_init_dag_linear (const void *sbuf, int sparts,
                                              int scount, ompi_datatype_t *sdtype,
                                              ompi_request_t ** srequest,
                                              void* rbuf, int rparts,
                                              int rcount, ompi_datatype_t *rdtype,
                                              ompi_request_t ** rrequest,
                                              struct ompi_communicator_t *comm,
                                              struct ompi_info_t *info,
                                              ompi_request_t ** request,
                                              struct mca_coll_base_module_2_4_0_t *module);
static mca_base_var_enum_value_t alltoall_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "pairwise"},
    {3, "binomial"},
    {0, NULL}
};
static mca_base_var_enum_value_t palltoallr_algorithms[] = {
    {0, "ignore"},
    {1, "partitioned_pt2pt"},
    {2, "bruck"},
    {3, "linear"},
#if OMPI_MPI_NOTIFICATIONS
    {4, "osc"},
#endif
    {5, "linear_partitions"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */
/* this routine is called by the component only */
int palltoallr_algorithm = 0;
int mca_coll_bullnbc_uniform_rddt = 0;
int ompi_coll_bullnbc_alltoall_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != alltoall_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ialltoall_algorithm_count",
                                         "Number of alltoall algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_alltoall_algorithms", alltoall_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ialltoall_algorithm",
                                         "Which alltoall algorithm is used unless MPI_IN_PLACE flag has been specified. If any, a specific algorithm will be used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ialltoall_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ialltoall algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);

  OBJ_RELEASE(new_enum);

  palltoallr_algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_palltoallr_algorithms", palltoallr_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "palltoallr_algorithm",
                                         "Which palltoallr algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &palltoallr_algorithm);
  OBJ_RELEASE(new_enum);

  /* TODO allgather to detect the  case of some coll does and some doesn't fit with this hint */
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "uniform_recv_datatypes",
                                  "Hint that all peers of a given palltoallr communication use the same datatype for reception. Enabled OSC optimizations",
                                  MCA_BASE_VAR_TYPE_BOOL, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_coll_bullnbc_uniform_rddt);

  return OMPI_SUCCESS;
}

/* simple linear MPI_Ialltoall the (simple) algorithm just sends to all nodes */
static int nbc_alltoall_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                             MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, p, res;
  MPI_Aint datasize;
  size_t sndsize;
  BULLNBC_Schedule *schedule;
  MPI_Aint rcvext, sndext;
  char inplace;
  enum {NBC_A2A_LINEAR, NBC_A2A_PAIRWISE, NBC_A2A_DISS, NBC_A2A_INPLACE} alg;
  void *tmpbuf = NULL;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
  ptrdiff_t gap = 0;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

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

  res = ompi_datatype_type_size(sendtype, &sndsize);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }

  if(inplace) {
    alg = NBC_A2A_INPLACE;
    goto selected_rule;
  }
  if(mca_coll_bullnbc_component.use_dynamic_rules) {
    if(0 != mca_coll_bullnbc_component.forced_params[ALLTOALL].algorithm) {
      alg = mca_coll_bullnbc_component.forced_params[ALLTOALL].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
      goto selected_rule;
    }
    if(bullnbc_module->com_rules[ALLTOALL]) {
      int algorithm,dummy1,dummy2,dummy3;
      algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[ALLTOALL],
                                                           sndsize * sendcount, &dummy1, &dummy2, &dummy3);
      if(algorithm) {
        alg = algorithm - 1;
        goto selected_rule;
      }
    }
  }
  /* algorithm selection */
  /* this number is optimized for TCP on odin.cs.indiana.edu */
  if (inplace) {
    alg = NBC_A2A_INPLACE;
  } else if (sndext * sendcount < 256){
      alg = NBC_A2A_DISS;
  } else {
    alg = NBC_A2A_PAIRWISE;
  }
#if 0
  size_t a2asize = sndsize * sendcount * p;
	  if((p <= 8) && ((a2asize < 1<<17) || (sndsize*sendcount < 1<<12))) {
    /* just send as fast as we can if we have less than 8 peers, if the
     * total communicated size is smaller than 1<<17 *and* if we don't
     * have eager messages (msgsize < 1<<13) */
    alg = NBC_A2A_LINEAR;
  } else if(a2asize < (1<<12)*(unsigned int)p) {
    /*alg = NBC_A2A_DISS;*/
    alg = NBC_A2A_LINEAR;
  } else
    alg = NBC_A2A_LINEAR; /*NBC_A2A_PAIRWISE;*/
#endif

selected_rule:
  opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                      "Bullnbc ialltoall : algorithm %d (no segmentation supported)",
                      alg + 1);
  /* allocate temp buffer if we need one */
  if (alg == NBC_A2A_INPLACE) {
    ptrdiff_t span = opal_datatype_span(&recvtype->super, recvcount, &gap);
    tmpbuf = malloc(span);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
  } else if (alg == NBC_A2A_DISS) {
    /* persistent operation is not supported currently for this algorithm */
    assert(! persistent);

    ompi_datatype_type_size(sendtype,&datasize);
    datasize *= sendcount;

    /* allocate temporary buffers */
    if ((p & 1) == 0) {
      tmpbuf = malloc (datasize * p * 2);
    } else {
      /* we cannot divide p by two, so alloc more to be safe ... */
      tmpbuf = malloc (datasize * (p / 2 + 1) * 2 * 2);
    }

    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* phase 1 - rotate n data blocks upwards into the tmpbuffer */
    uint64_t flags;
    int is_accel_buf1, is_accel_buf2;
    int dev_id;
    is_accel_buf1 = opal_accelerator.check_addr(sendbuf, &dev_id, &flags);
    is_accel_buf2 = opal_accelerator.check_addr(recvbuf, &dev_id, &flags);
    if (NBC_Type_intrinsic(sendtype) && !is_accel_buf1 && !is_accel_buf2 ) {
      /* contiguous - just copy (1st copy) */
      memcpy (tmpbuf, (char *) sendbuf + datasize * rank, datasize * (p - rank));
      if (rank != 0) {
        memcpy ((char *) tmpbuf + datasize * (p - rank), sendbuf, datasize * rank);
      }
    } else {
      /* non-contiguous - pack */
      res = ompi_datatype_sndrcv(((const char *) sendbuf) + rank * sendcount *sndext,
                           (p - rank) * sendcount, sendtype,
                           tmpbuf, (p - rank) * (int)datasize, MPI_BYTE);

      if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
        NBC_Error("MPI Error in ompi_datatype_sndrcv() (%i)", res);
        free(tmpbuf);
        return res;
      }

      if (rank != 0) {
        res = ompi_datatype_sndrcv(sendbuf, rank * sendcount, sendtype,
                             ((char *) tmpbuf) + datasize * (p - rank), rank * (int)datasize, MPI_BYTE);
        if (OPAL_UNLIKELY(MPI_SUCCESS != res)) {
          NBC_Error("MPI Error in ompi_datatype_sndrcv() (%i)", res);
          free(tmpbuf);
          return res;
        }
      }
    }
  }

    /* not found - generate new schedule */
    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      free(tmpbuf);
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (!inplace) {
      /* copy my data to receive buffer */
      char *rbuf = (char *) recvbuf + (MPI_Aint)rank * (MPI_Aint)recvcount * rcvext;
      char *sbuf = (char *) sendbuf + (MPI_Aint)rank * (MPI_Aint)sendcount * sndext;
      res = NBC_Sched_copy (sbuf, false, sendcount, sendtype,
                            rbuf, false, recvcount, recvtype, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
    }

    switch(alg) {
      case NBC_A2A_INPLACE:
        res = a2a_sched_inplace(rank, p, schedule, recvbuf, recvcount, recvtype, rcvext, gap, comm);
        break;
      case NBC_A2A_LINEAR:
        res = a2a_sched_linear(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
      case NBC_A2A_DISS:
        res = a2a_sched_diss(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm, tmpbuf);
        break;
      case NBC_A2A_PAIRWISE:
        res = a2a_sched_pairwise(rank, p, sndext, rcvext, schedule, sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
        break;
    }

    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    res = NBC_Sched_commit(schedule);
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

int ompi_coll_bullnbc_ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                               MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
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

static int nbc_alltoall_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int res, rsize;
  MPI_Aint sndext, rcvext;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent (sendtype, &sndext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  res = ompi_datatype_type_extent (recvtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  for (int i = 0; i < rsize; i++) {
    char *rbuf, *sbuf;
    /* post all sends */
    sbuf = (char *) sendbuf + i * sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, i, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      break;
    }

    /* post all receives */
    rbuf = (char *) recvbuf + i * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      break;
    }
  }

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

  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_ialltoall_inter (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
				      MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
				      struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoall_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
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

static inline int a2a_sched_pairwise(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule* schedule,
                                     const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                     MPI_Datatype recvtype, MPI_Comm comm) {

  if (p < 2) {
    return OMPI_SUCCESS;
  }

  for (int r = 1 ; r < p ; ++r) {
    int res;
    int sndpeer = (rank + r) % p;
    int rcvpeer = (rank - r + p) % p;

    char *rbuf = (char *) recvbuf + rcvpeer * recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, rcvpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    char *sbuf = (char *) sendbuf + sndpeer * sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, sndpeer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_linear(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule* schedule,
                                   const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                   MPI_Datatype recvtype, MPI_Comm comm) {
  int res;

  for (int r = 0 ; r < p ; ++r) {
    /* easy algorithm */
    if (r == rank) {
      continue;
    }

    char *rbuf = (char *) recvbuf + (intptr_t)r * (intptr_t)recvcount * rcvext;
    res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, r, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    char *sbuf = (char *) sendbuf + (intptr_t)r * (intptr_t)sendcount * sndext;
    res = NBC_Sched_send (sbuf, false, sendcount, sendtype, r, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_diss(int rank, int p, MPI_Aint sndext, MPI_Aint rcvext, BULLNBC_Schedule* schedule,
                                 const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                 MPI_Datatype recvtype, MPI_Comm comm, void* tmpbuf) {
  int res;
  MPI_Aint datasize, offset;
  char *rtmpbuf, *stmpbuf;

  if (p < 2) {
    return OMPI_SUCCESS;
  }

  ompi_datatype_type_size(sendtype,&datasize);
  datasize *= sendcount;

  /* allocate temporary buffers */
  if ((p & 1) == 0) {
    rtmpbuf = (char *)tmpbuf + datasize * p;
    stmpbuf = (char *)tmpbuf + datasize * (p + p / 2);
  } else {
    /* we cannot divide p by two, so alloc more to be safe ... */
    int virtp = (p / 2 + 1) * 2;
    rtmpbuf = (char *)tmpbuf + datasize * p;
    stmpbuf = (char *)tmpbuf + datasize * (p + virtp / 2);
  }

  /* phase 2 - communicate */
  for (int r = 1; r < p; r <<= 1) {
    offset = 0;
    for (int i = 1 ; i < p; ++i) {
      /* test if bit r is set in rank number i */
      if (i & r) {
        /* copy data to sendbuffer (2nd copy) - could be avoided using iovecs */
        /*printf("[%i] round %i: copying element %i to buffer %lu\n", rank, r, i, (unsigned long)(stmpbuf+offset));*/
        res = NBC_Sched_copy((void *)(intptr_t)(i * datasize), true, datasize, MPI_BYTE, stmpbuf + offset -
                             (intptr_t)tmpbuf, true, datasize, MPI_BYTE, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
        offset += datasize;
      }
    }

    int speer = (rank + r) % p;
    /* add p because modulo does not work with negative values */
    int rpeer = ((rank - r) + p) % p;

    res = NBC_Sched_recv (rtmpbuf - (intptr_t)tmpbuf, true, offset, MPI_BYTE, rpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    res = NBC_Sched_send (stmpbuf - (intptr_t)tmpbuf, true, offset, MPI_BYTE, speer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    /* unpack from buffer */
    offset = 0;
    for (int i = 1; i < p; ++i) {
      /* test if bit r is set in rank number i */
      if (i & r) {
        /* copy data to tmpbuffer (3rd copy) - could be avoided using iovecs */
        res = NBC_Sched_copy (rtmpbuf + offset - (intptr_t)tmpbuf, true, datasize, MPI_BYTE,
                              (void *)(intptr_t)(i * datasize), true, datasize, MPI_BYTE, schedule,
                              false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }

        offset += datasize;
      }
    }
  }

  /* phase 3 - reorder - data is now in wrong order in tmpbuf - reorder it into recvbuf */
  for (int i = 0 ; i < p; ++i) {
    char* rbuf = (char *) recvbuf + ((rank - i + p) % p) * recvcount * rcvext;
    res = NBC_Sched_unpack ((void *)(intptr_t) (i * datasize), true, recvcount, recvtype, rbuf, false, schedule,
                            false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int a2a_sched_inplace(int rank, int p, BULLNBC_Schedule* schedule, void* buf, int count,
                                   MPI_Datatype type, MPI_Aint ext, ptrdiff_t gap, MPI_Comm comm) {
  int res;

  for (int i = 1 ; i < (p+1)/2 ; i++) {
    int speer = (rank + i) % p;
    int rpeer = (rank + p - i) % p;
    char *sbuf = (char *) buf + (intptr_t)speer * (intptr_t)count * ext;
    char *rbuf = (char *) buf + (intptr_t)rpeer * (intptr_t)count * ext;

    res = NBC_Sched_copy (rbuf, false, count, type,
                          (void *)(-gap), true, count, type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_send (sbuf, false , count, type, speer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (rbuf, false , count, type, rpeer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    res = NBC_Sched_send ((void *)(-gap), true, count, type, rpeer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (sbuf, false, count, type, speer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }
  if (0 == (p%2)) {
    int peer = (rank + p/2) % p;

    char *tbuf = (char *) buf + (intptr_t)peer * (intptr_t)count * ext;
    res = NBC_Sched_copy (tbuf, false, count, type,
                          (void *)(-gap), true, count, type,
                          schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_send ((void *)(-gap), true , count, type, peer, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
    res = NBC_Sched_recv (tbuf, false , count, type, peer, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_alltoall_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module) {

    int res = nbc_alltoall_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_alltoall_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_alltoall_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
                                      comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_palltoall_init (const void *sbuf, int sparts, int scount, struct ompi_datatype_t *sdtype,
                                      void* rbuf, int rparts, int rcount, struct ompi_datatype_t *rdtype,
                                      struct ompi_communicator_t *comm, struct ompi_info_t *info,
                                      ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module)
{
    return ompi_coll_bullnbc_palltoallr_init_select( sbuf, sparts, scount, sdtype,
                                                    MPIX_NO_REQUESTS,
                                                    rbuf, rparts, rcount, rdtype,
                                                    MPIX_NO_REQUESTS,
                                                    comm, info, request, module);
}

OBJ_CLASS_INSTANCE(pcoll_dag_item, opal_free_list_item_t, NULL, NULL);



/* Send and receive data without taking care of partition subdivision,
 * in a Bruck pattern: perform log2(commsize) iterations and use peers as
 * intermediary for data transfers. This reduce the number of communication but
 * the global amount of data transfered.
 * This algorithm targets small message sizes.
 *
 * Ranks are implicitly renumered relatively to local rank
 * At iteration 1, data for peers 1, 11, 101, 111, 1001, ... are sent to rank +1
 * At iteration 2, data for peers 10, 11, 110, 111, 1010,... are sent to rank +2
 * At iteration 3, data for peers 100, 101, 110, 111, 1100 ... are sent to rank +4
 * ...
 * At the end data are arrieved and can be placed in receive buffer.
 */
static int
ompi_coll_bullnbc_palltoallr_init_bruck(const void *sbuf, int sparts,
                                        int scount, ompi_datatype_t *sdtype,
                                        ompi_request_t ** srequest,
                                        void* rbuf, int rparts,
                                        int rcount, ompi_datatype_t *rdtype,
                                        ompi_request_t ** rrequest,
                                        struct ompi_communicator_t *comm,
                                        struct ompi_info_t *info,
                                        ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module)
{
    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;

    MPI_Aint rcvext, sndext;
    int ret = OMPI_SUCCESS, comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    size_t ssize, rsize, blocksize;
    ompi_datatype_type_size(sdtype,&ssize);
    ompi_datatype_type_size(rdtype,&rsize);
    blocksize = sparts * scount * ssize;

    int n_iter = 0;
    for (int mask = comm_size -1; mask > 0; mask /=2) {
        ++ n_iter;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sparts * comm_size, srequest,
                                                   rparts * comm_size, rrequest,
                                                   2*n_iter, n_iter,
                                                   comm, request);


    /* tmpbuf for 1 iteration (recv buffers) */
    size_t iter_tmpbuf_size = comm_size /2 * blocksize;
    size_t tmpbuf_size = iter_tmpbuf_size *(1 + n_iter); /* start + n_iter */
    /* tmpbuf not used in case of a single iteration: recv targets rbuf */
    req->tmpbuf = (n_iter <= 1) ? NULL: bullnbc_xmalloc (tmpbuf_size);



    /* First iteration using these blocks: to know the send/recv
     * to bind with parts completion */
    int *first_iter_uses = bullnbc_xmalloc(comm_size * sizeof(int));
    int *n_first = bullnbc_xmalloc(n_iter * sizeof(int));
    for (int iter = 0; iter < n_iter; ++iter){
        n_first[iter] = 0;
    }
    for (int vpeer = 1; vpeer < comm_size; ++vpeer){
        int iter = 0;
        int mask = 1;
        while ( !(vpeer & mask) ){
            mask *=2;
            ++iter;
        }
        first_iter_uses[vpeer] = iter;
        n_first[iter] ++;
    }

    /* History of data move for each iteration */
    /* Store offset in tmpbuf for data from each peer at each iteration */
    char ** block_pos = bullnbc_xmalloc( (1+n_iter) * comm_size * sizeof(char*));
    for (int vpeer=0; vpeer < comm_size; ++vpeer) {
        block_pos[vpeer] = ((char*)sbuf) + ((rank+vpeer)% comm_size) *blocksize;
    }
    for (int mask = 1, iter = 1; mask < comm_size; mask *=2, ++iter) {
        int offset = 0;
        for (int vpeer=0; vpeer < comm_size; ++vpeer) {
            int vpeer_idx = comm_size * iter + vpeer;
            if(vpeer & mask){
                /* Block received at this iteration */
                block_pos[vpeer_idx] = req->tmpbuf
                                       + iter*iter_tmpbuf_size
                                       + offset*blocksize;
                offset ++;
            } else {
                block_pos[vpeer_idx] = block_pos[comm_size * (iter -1) + vpeer];
            }
        }
    }

    /* Local block */
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

    pcoll_dag_item* pack = sched_convertion(((char*)sbuf) + rank*sparts*scount*sndext,
                                      sparts*scount, sdtype,
                                      ((char*)rbuf) + rank*rparts*rcount*rcvext,
                                      rparts*rcount, rdtype, sparts + rparts);
    for (int part=0; part < sparts; ++part){
        pcoll_dag_item* complete = sched_complete(sparts*rank + part, req);
        create_dependency(pack, complete);
    }
    for (int part=0; part < rparts; ++part){
        pcoll_dag_item* arrived = sched_complete(req->total_sparts + rparts*rank + part, req);
        create_dependency(pack, arrived);
    }



    /* Other blocks: prepare schedule per iterations */
    ptrdiff_t * disps = bullnbc_xmalloc(comm_size * sizeof(int));
    int * lengths = bullnbc_xmalloc(comm_size * sizeof(int));
    MPI_Datatype* ddts = bullnbc_xmalloc(comm_size * sizeof(int));
    pcoll_dag_item** complete_on_send = bullnbc_xmalloc(sparts*comm_size * sizeof(pcoll_dag_item*));

    req->n_created_ddt = 2*n_iter;
    req->created_ddt = bullnbc_xmalloc (2*n_iter * sizeof(ompi_datatype_t*));
    pcoll_dag_item ** send_items = bullnbc_xmalloc(n_iter * sizeof(pcoll_dag_item*));

    for (int mask = 1, iter = 0; mask < comm_size; mask *=2, ++iter) {

        /* prepare the derived datatype for the send of this iteration */
        int offset = 0;
        int n_first_use = 0;
        for (int vrank=1; vrank<comm_size; ++vrank) {
            if (0 == (vrank & mask)){
                continue;
            }
            int peer = (rank + vrank) % comm_size;

            disps[offset] = block_pos[iter*comm_size + vrank] - req->tmpbuf;
            /* datatype depends if it is the first use of the block */
            if (iter == first_iter_uses[vrank]) {
                ddts[offset] = sdtype;
                lengths[offset] = sparts*scount;
                for (int part=0; part < sparts; ++part){
                    pcoll_dag_item* complete = sched_complete(sparts*peer + part, req);
                    complete_on_send[n_first_use*sparts + part] = complete;
                }
                n_first_use ++;
            } else {
                ddts[offset] = MPI_BYTE;
                lengths[offset] = blocksize;
            }
            offset ++;
        }
        struct ompi_datatype_t * ddt;
        ompi_datatype_create_struct(offset, lengths, disps, ddts, &ddt);
        ompi_datatype_commit(&ddt);
        req->created_ddt[iter] = ddt;

        NBC_DEBUG(5," send %d %p of %d byte from rank %d\n",
                  mask, send, offset*blocksize,
                  (rank + mask) % comm_size);

        pcoll_dag_item* send = sched_send(req->tmpbuf, 1, ddt,
                                          (rank + mask) % comm_size, comm,
                                          &req->internal_reqs[iter], req,
                                          sparts *n_first[iter]);
        for (int i = 0; i < n_first_use; ++i) {
            for (int part=0; part < sparts; ++part){
                create_dependency(send, complete_on_send[i*sparts + part]);
            }
        }
        send_items[iter] = send;
    }
    free(complete_on_send);

    /* Pready entry points attached to packing/send schedule */
    for (int peer=0; peer < comm_size; ++peer) {
        int vrank  = (peer - rank + comm_size)% comm_size;

        pcoll_dag_item* next;
        if (vrank) {
            next = send_items[first_iter_uses[vrank]];
        } else {
            next = pack;
        }

        for (int part=0; part < sparts; ++part){
            pcoll_dag_item* ready = sched_ready(peer*sparts + part, req, 1);
            create_dependency(ready, next);
        }
    }

    pcoll_dag_item** complete_on_recv = bullnbc_xmalloc((n_iter+rparts*comm_size) * sizeof(pcoll_dag_item*));

    /* Schedule recv for each iteration */
    for (int mask = 1, iter = 0; mask < comm_size; mask *=2, ++iter) {

        int offset = 0;
        int deps_idx = 0;

        /* Link recv to sends for next iterations */
        for (int next_iter = iter +1, next_mask = mask*2;
             next_mask + mask < comm_size;
             next_mask *= 2, next_iter ++){
            complete_on_recv[deps_idx++] = send_items[next_iter];
        }

        /* Prepare recv derived datatype to unpack in user buffer at the end */
        for (int vrank=1; vrank<comm_size; ++vrank) {
            if (0 == (vrank & mask)){
                continue;
            }

            if (vrank < 2* mask) {
                /* Last use of the block in comms : unpack it in rbuf */
                int peer = (rank - vrank + comm_size)%comm_size;
                ddts[offset] = rdtype;
                lengths[offset] = rparts*rcount;
                disps[offset] =  ((char*) rbuf) + peer*rcvext*rcount*rparts - req->tmpbuf;

                NBC_DEBUG(5, "sched recv %d %p block%d (%dx%d elem) at %p\n",
                          mask, recv, offset, rparts, rcount, disps[offset]+req->tmpbuf);
                /* Link recv to arrived */
                for (int part=0; part < rparts; ++part){
                    pcoll_dag_item* arrived = sched_complete(req->total_sparts
                                                             + rparts*peer + part,
                                                             req);
                    complete_on_recv[deps_idx++] = arrived;
                }
            } else {
                /* This block must be sent at least once more */
                ddts[offset] = MPI_BYTE;
                lengths[offset] = blocksize;
                disps[offset] =  block_pos[comm_size * (iter+1) + vrank] - req->tmpbuf;
                NBC_DEBUG(5, "sched recv %d %p block%d at %p\n",
                          mask, recv, offset, disps[offset]+req->tmpbuf);
            }
            offset ++;
        }
        struct ompi_datatype_t * ddt;
        ompi_datatype_create_struct(offset, lengths, disps, ddts, &ddt);
        ompi_datatype_commit(&ddt);
        req->created_ddt[n_iter + iter] = ddt;

        pcoll_dag_item* recv = sched_recv(req->tmpbuf, 1, ddt,
                                          (rank - mask + comm_size) % comm_size, comm,
                                          &req->internal_reqs[iter + n_iter], req,
                                          req->n_ready_nodes + iter,
                                          deps_idx);

        NBC_DEBUG(5, "sched Recv %d %p (%d blocks) with req %p\n", mask, recv, offset, recv->args.comm.req);
        for (int i = 0; i < deps_idx; ++i){
            create_dependency(recv, complete_on_recv[i]);
        }
    }
    free(complete_on_recv);
    free(disps);
    free(lengths);
    free(ddts);

    free(send_items);
    free(block_pos);
    free(first_iter_uses);
    free(n_first);

    return OMPI_SUCCESS;
}

/* Schedule a send and a receive for each peer */
static int
ompi_coll_bullnbc_palltoallr_init_dag_linear( const void *sbuf, int sparts,
                                              int scount, ompi_datatype_t *sdtype,
                                              ompi_request_t ** srequest,
                                              void* rbuf, int rparts,
                                              int rcount, ompi_datatype_t *rdtype,
                                              ompi_request_t ** rrequest,
                                              struct ompi_communicator_t *comm,
                                              struct ompi_info_t *info,
                                              ompi_request_t ** request,
                                              struct mca_coll_base_module_2_4_0_t *module)
{
    int ret = OMPI_SUCCESS, comm_size, rank;
    MPI_Aint rcvext, sndext;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sparts * comm_size, srequest,
                                                   rparts * comm_size, rrequest,
                                                   2 * (comm_size -1), comm_size -1,
                                                   comm, request);

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

    int offset=0;
    for ( int peer = 0 ; peer < comm_size; ++peer){
        if (peer == rank) {
            offset = -1;
            continue;
        }

        pcoll_dag_item* recv = sched_recv(((char*)rbuf) + peer*rparts*rcount*rcvext,
                                          rcount*rparts, rdtype,
                                          peer, comm,
                                          &req->internal_reqs[peer + offset], req,
                                          req->n_ready_nodes + peer+offset,
                                          rparts);
        for (int part=0; part < rparts; ++part){
            pcoll_dag_item* complete = sched_complete(req->total_sparts + rparts*peer + part, req);
            create_dependency(recv, complete);
        }
        NBC_DEBUG(1, "Sched send to %d\n", peer);
        pcoll_dag_item* send = sched_send(((char*)sbuf) + peer*sparts*scount*sndext,
                                          scount*sparts, sdtype,
                                          peer, comm,
                                          &req->internal_reqs[comm_size -1 + peer+offset], req,
                                          sparts);
        for (int part=0; part < sparts; ++part){
            pcoll_dag_item* ready = sched_ready(sparts*peer + part, req, 1);
            create_dependency(ready, send);
            pcoll_dag_item* complete = sched_complete(sparts*peer + part, req);
            create_dependency(send, complete);
        }
    }


    /* Local block */
    pcoll_dag_item* pack = sched_convertion(((char*)sbuf) + rank*sparts*scount*sndext,
                                      sparts*scount, sdtype,
                                      ((char*)rbuf) + rank*rparts*rcount*rcvext,
                                      rparts*rcount, rdtype, sparts + rparts);
    for (int part=0; part < sparts; ++part){
        pcoll_dag_item* ready = sched_ready(sparts*rank + part, req, 1);
        create_dependency(ready, pack);
        pcoll_dag_item* complete = sched_complete(sparts*rank + part, req);
        create_dependency(pack, complete);
    }
    for (int part=0; part < rparts; ++part){
        pcoll_dag_item* arrived = sched_complete(req->total_sparts + rparts*rank + part, req);
        create_dependency(pack, arrived);
    }

    return OMPI_SUCCESS;
}

/* Schedule a send and a receive for each partition */
static int
ompi_coll_bullnbc_palltoallr_init_dag_linear_parts( const void *sbuf,
                                                    int sparts, int scount,
                                                    ompi_datatype_t *sdtype,
                                                    ompi_request_t ** srequest,
                                                    void* rbuf,
                                                    int rparts, int rcount,
                                                    ompi_datatype_t *rdtype,
                                                    ompi_request_t ** rrequest,
                                                    struct ompi_communicator_t *comm,
                                                    struct ompi_info_t *info,
                                                    ompi_request_t ** request,
                                                    struct mca_coll_base_module_2_4_0_t *module)
{
    int ret = OMPI_SUCCESS, comm_size, rank;
    MPI_Aint rcvext, sndext;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sparts * comm_size, srequest,
                                                   rparts * comm_size, rrequest,
                                                   2 * (comm_size -1) * rparts,
                                                   (comm_size -1) * rparts,
                                                   comm, request);
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

    /* Requests for internal true comms */
    int offset=0;
    for ( int peer = 0 ; peer < comm_size; ++peer){
        if (peer == rank) {
            offset = -1;
            continue;
        }

        for (int part=0; part < rparts; ++part){
            pcoll_dag_item* recv = sched_recv(((char*)rbuf) + (peer*rparts + part)*rcount*rcvext,
                                              rcount, rdtype,
                                              peer, comm,
                                              &req->internal_reqs[(peer+offset)*rparts + part], req,
                                              req->n_ready_nodes + (peer+offset)*rparts+part,
                                              1);
            pcoll_dag_item* complete = sched_complete(req->total_sparts + rparts*peer + part, req);
            create_dependency(recv, complete);
        }

        int internal_req_shift = (comm_size -1)*rparts + (peer+offset)*sparts;
        for (int part=0; part < sparts; ++part){
            int internal_req_idx = internal_req_shift + part;
            pcoll_dag_item* send = sched_send(((char*)sbuf) + (peer*sparts + part)*scount*sndext,
                                              scount, sdtype,
                                              peer, comm,
                                              &req->internal_reqs[internal_req_idx], req,
                                              1);
            pcoll_dag_item* ready = sched_ready(sparts*peer + part, req, 1);
            create_dependency(ready, send);
            pcoll_dag_item* complete = sched_complete(sparts*peer + part, req);
            create_dependency(send, complete);
        }
    }


    /* Local block */
    pcoll_dag_item* pack = sched_convertion(((char*)sbuf) + rank*sparts*scount*sndext,
                                      sparts*scount, sdtype,
                                      ((char*)rbuf) + rank*rparts*rcount*rcvext,
                                      rparts*rcount, rdtype, sparts + rparts);
    for (int part=0; part < sparts; ++part){
        pcoll_dag_item* ready = sched_ready(sparts*rank + part, req, 1);
        create_dependency(ready, pack);
        pcoll_dag_item* complete = sched_complete(sparts*rank + part, req);
        create_dependency(pack, complete);
    }
    for (int part=0; part < rparts; ++part){
        pcoll_dag_item* arrived = sched_complete(req->total_sparts + rparts*rank + part, req);
        create_dependency(pack, arrived);
    }

    return OMPI_SUCCESS;
}

#if OMPI_MPI_NOTIFICATIONS
/* Schedule a notified put and a notification wait for each peer */
static int
ompi_coll_bullnbc_palltoallr_init_dag_osc( const void *sbuf, int sparts,
                                           int scount, ompi_datatype_t *sdtype,
                                           ompi_request_t ** srequest,
                                           void* rbuf, int rparts,
                                           int rcount, ompi_datatype_t *rdtype,
                                           ompi_request_t ** rrequest,
                                           struct ompi_communicator_t *comm,
                                           struct ompi_info_t *info,
                                           ompi_request_t ** request,
                                           struct mca_coll_base_module_2_4_0_t *module)
{
    int ret = OMPI_SUCCESS, comm_size, rank;
    MPI_Aint rcvext, sndext;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sparts * comm_size, srequest,
                                                   rparts * comm_size, rrequest,
                                                   0, comm_size -1,
                                                   comm, request);

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

    NBC_DEBUG(10, "OSC alltoallr create win of size %dx%d elem of size %d\n", rcount, rparts, rcvext);
    ret = ompi_win_create_notify(rbuf, rcvext*rcount*rparts*comm_size, rcvext, comm,
                                 &info->super, &req->win);
    if (ret != OMPI_SUCCESS) {
        NBC_Error("Error %d creating attached window", ret);
        abort();
    }

    int offset=0;
    for ( int peer = 0 ; peer < comm_size; ++peer){
        if (peer == rank) {
            offset = -1;
            continue;
        }
        pcoll_dag_item* notif = sched_notif(req,
                                            peer,
                                            req->n_ready_nodes + peer + offset,
                                            rparts);
        for (int part=0; part < rparts; ++part){
            pcoll_dag_item* complete = sched_complete(req->total_sparts + rparts*peer + part, req);
            create_dependency(notif, complete);
        }
        pcoll_dag_item* put = sched_put(((char*)sbuf) + peer*sparts*scount*sndext,
                                        scount*sparts, sdtype,
                                        peer, rcount*rparts*rank,
                                        rcount*rparts, rdtype,
                                        req, rank,
                                        sparts);
        for (int part=0; part < sparts; ++part){
            pcoll_dag_item* ready = sched_ready(sparts*peer + part, req, 1);
            create_dependency(ready, put);
            pcoll_dag_item* complete = sched_complete(sparts*peer + part, req);
            create_dependency(put, complete);
        }
    }


    /* Local block */
    pcoll_dag_item* pack = sched_convertion(((char*)sbuf) + rank*sparts*scount*sndext,
                                      sparts*scount, sdtype,
                                      ((char*)rbuf) + rank*rparts*rcount*rcvext,
                                      rparts*rcount, rdtype, sparts + rparts);
    for (int part=0; part < sparts; ++part){
        pcoll_dag_item* ready = sched_ready(sparts*rank + part, req, 1);
        create_dependency(ready, pack);
        pcoll_dag_item* complete = sched_complete(sparts*rank + part, req);
        create_dependency(pack, complete);
    }
    for (int part=0; part < rparts; ++part){
        pcoll_dag_item* arrived = sched_complete(req->total_sparts + rparts*rank + part, req);
        create_dependency(pack, arrived);
    }

    return OMPI_SUCCESS;
}
#endif /* OMPI_MPI_NOTIFICATIONS */

/*  Schedule a partitioned send/recv for each peer */
static int
ompi_coll_bullnbc_palltoallr_part_pt2pt (const void *sbuf, int sparts,
                                         int scount, struct ompi_datatype_t *sdtype,
                                         ompi_request_t ** srequest,
                                         void* rbuf, int rparts,
                                         int rcount, struct ompi_datatype_t *rdtype,
                                         ompi_request_t ** rrequest,
                                         struct ompi_communicator_t *comm,
                                         struct ompi_info_t *info,
                                         ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module)
{
    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;

    MPI_Aint rcvext, sndext;
    ompi_request_t **psend_req, **precv_req;
    int ret = OMPI_SUCCESS, comm_size;
    comm_size = ompi_comm_size (comm);


    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_alloc_pcoll_request();

    req->comm = comm;
    req->tag = ompi_coll_base_nbc_reserve_tags(comm, 1);
    req->user_sreqs = srequest;
    req->user_rreqs = rrequest;

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

    /* Prepare list of subpart for each user exposed requests */
    req->total_sparts = sparts * comm_size;
    req->total_rparts = rparts * comm_size;
    req->send_subparts = bullnbc_xmalloc(req->total_sparts * sizeof(opal_list_t));
    req->recv_subparts = bullnbc_xmalloc(req->total_rparts * sizeof(opal_list_t));

    for (size_t part=0; part<req->total_sparts; ++part){
        OBJ_CONSTRUCT(&req->send_subparts[part], opal_list_t);
    }
    for (size_t part=0; part<req->total_rparts; ++part){
        OBJ_CONSTRUCT(&req->recv_subparts[part], opal_list_t);
    }

    /* Prepare internal psend/precv requests */
    req->n_internal_reqs = 2*comm_size;
    req->internal_reqs = bullnbc_xmalloc(req->n_internal_reqs * sizeof(ompi_request_t *));

    precv_req = req->internal_reqs;
    psend_req = req->internal_reqs + comm_size;

    for(int i = 0 ; i < comm_size ; ++i){
        NBC_DEBUG(20,"Init recv at %p and send and %p\n",
                  ((char*)rbuf) + i*rparts*rcount*rcvext,
                ((char*)sbuf) + i*sparts*scount*sndext);

        mca_part.part_precv_init(((char*)rbuf) + i*rparts*rcount*rcvext,
                                 rparts, rcount, rdtype, i,
                                 req->tag, comm, info,
                                 &precv_req[i]);

        /* Append internal precv partition to the respective subpart list*/
        for(int part = 0 ; part < rparts ; ++part){
            ompi_coll_bullnbc_subpart* subpart;
            subpart = (ompi_coll_bullnbc_subpart*) opal_free_list_wait(&subpart_free_list);
            subpart->sub_req = precv_req[i];
            subpart->part_idx = part;
            opal_list_append(&req->recv_subparts[i*rparts + part],
                             &subpart->super.super);
        }

        mca_part.part_psend_init(((char*)sbuf) + i*sparts*scount*sndext,
                                 sparts, scount, sdtype, i,
                                 req->tag, comm, info,
                                 &psend_req[i]);

        for(int part = 0 ; part < sparts ; ++part){
            ompi_coll_bullnbc_subpart* subpart;
            subpart = (ompi_coll_bullnbc_subpart*) opal_free_list_wait(&subpart_free_list);
            subpart->sub_req = psend_req[i];
            subpart->part_idx = part;
            opal_list_append(&req->send_subparts[i*sparts + part],
                             &subpart->super.super);
        }
    }

    /* Init user exposed request and  bind to the list of subparts */
    subpart_requests_create(req);

    *request = &req->req_ompi;
    return OMPI_SUCCESS;
}

int
ompi_coll_bullnbc_palltoallr_init_select(const void *sbuf,
                                         int sparts, int scount,
                                         struct ompi_datatype_t *sdtype,
                                         ompi_request_t ** srequest,
                                         void* rbuf,
                                         int rparts, int rcount,
                                         struct ompi_datatype_t *rdtype,
                                         ompi_request_t ** rrequest,
                                         struct ompi_communicator_t *comm,
                                         struct ompi_info_t *info,
                                         ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module)
{
    size_t total_size;

    char inplace;
    NBC_IN_PLACE(sbuf, rbuf, inplace);
    if (inplace && rcount) {
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "No MPI_IN_PLACE for partitioned", true,
                       "palltoallr");
        ompi_mpi_abort(comm, 0);
    }

    ompi_datatype_type_size(rdtype, &total_size);
    total_size *= rcount*rparts;

    /* TODO total_size == 0 */

    if (! mca_coll_bullnbc_uniform_rddt && palltoallr_algorithm == 4){
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Non uniform partitionned, no rma", true,
                       "palltoallr");
        palltoallr_algorithm = 0;
    }

    int alg = palltoallr_algorithm;
    if (0 == alg) {
        if (total_size < 1024){
            alg = 2;
        } else {
            alg = 3;
        }
    }

    if (! mca_coll_bullnbc_uniform_partition_sizes && 5 == alg){
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Non uniform partition sizes", true,
                       5, "alltoall", 3);
        alg = 3;
    }

    module_need_progress((ompi_coll_bullnbc_module_t*) module);

    switch(alg){
        case 1:
            return ompi_coll_bullnbc_palltoallr_part_pt2pt(sbuf, sparts, scount, sdtype, srequest, rbuf, rparts, rcount, rdtype, rrequest, comm, info, request, module);
        case 2:
            return ompi_coll_bullnbc_palltoallr_init_bruck(sbuf, sparts, scount, sdtype, srequest, rbuf, rparts, rcount, rdtype, rrequest, comm, info, request, module);
       case 3:
            return ompi_coll_bullnbc_palltoallr_init_dag_linear(sbuf, sparts, scount, sdtype, srequest, rbuf, rparts, rcount, rdtype, rrequest, comm, info, request, module);
#if OMPI_MPI_NOTIFICATIONS
       case 4:
            return ompi_coll_bullnbc_palltoallr_init_dag_osc(sbuf, sparts, scount, sdtype, srequest, rbuf, rparts, rcount, rdtype, rrequest, comm, info, request, module);
#endif /* OMPI_MPI_NOTIFICATIONS */
        case 5:
            return ompi_coll_bullnbc_palltoallr_init_dag_linear_parts(sbuf, sparts, scount, sdtype, srequest, rbuf, rparts, rcount, rdtype, rrequest, comm, info, request, module);
        default:
            fprintf(stderr, "Palltoallr algorithm %d is not implemented yet\n", alg); fflush(stderr);
            abort();
    }
}
