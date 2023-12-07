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
 * Copyright (c) 2015      Mellanox Technologies. All rights reserved.
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
#include "base/coll_bullnbc_base.h"

static const mca_base_var_enum_value_t barrier_algorithms[] = {
    {0, "legacy"},
    {1, "linear"},
    {2, "double_ring"},
    {3, "recursive_doubling"},
    {4, "bruck"},
    {5, "two_proc"},
    {6, "tree"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */
/* this routine is called by the component only */

int ompi_coll_bullnbc_barrier_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != barrier_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ibarrier_algorithm_count",
                                         "Number of barrier algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_barrier_algorithms", barrier_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ibarrier_algorithm",
                                         "Which barrier algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ibarrier_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ibarrier algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

/* Dissemination implementation of MPI_Ibarrier */
static int coll_bullnbc_ibarrier_legacy(struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, p, maxround, res;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    maxround = ceil_of_log2(p) -1;

    for (int round = 0 ; round <= maxround ; ++round) {
      int sendpeer = (rank + (1 << round)) % p;
      /* add p because modulo does not work with negative values */
      int recvpeer = ((rank - (1 << round)) + p) % p;

      /* send msg to sendpeer */
      res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, sendpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }

      /* recv msg from recvpeer */
      res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, recvpeer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }

      /* end communication round */
      if (round < maxround) {
        res = NBC_Sched_barrier (schedule);
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

  return OMPI_SUCCESS;
}

static int nbc_barrier_init(struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
    int algorithm = 0;
    const int p = ompi_comm_size(comm);

    if(mca_coll_bullnbc_component.use_dynamic_rules) {
        ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
        algorithm = mca_coll_bullnbc_component.forced_params[BARRIER].algorithm;

        if(algorithm == 0 && bullnbc_module->com_rules[BARRIER]) {
            int dummy1, dummy2, dummy3;
            algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[BARRIER],
                        0, &dummy1, &dummy2, &dummy3);
        }
    }

    if (p != 2 && algorithm == 5) {
         algorithm = 0;
    }

    switch(algorithm) {
    case (0):   return coll_bullnbc_ibarrier_legacy(comm, request, module, persistent);
    case (1):   return ompi_coll_bullnbc_base_ibarrier_intra_basic_linear(comm, request, module, persistent);
    case (2):   return ompi_coll_bullnbc_base_ibarrier_intra_doublering(comm, request, module, persistent);
    case (3):   return ompi_coll_bullnbc_base_ibarrier_intra_recursivedoubling(comm, request, module, persistent);
    case (4):   return ompi_coll_bullnbc_base_ibarrier_intra_bruck(comm, request, module, persistent);
    case (5):   return ompi_coll_bullnbc_base_ibarrier_intra_two_procs(comm, request, module, persistent);
    case (6):   return ompi_coll_bullnbc_base_ibarrier_intra_tree(comm, request, module, persistent);
    }

    return OMPI_ERROR;
}

int ompi_coll_bullnbc_ibarrier(struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_barrier_init(comm, request, module, false);
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

static int nbc_barrier_inter_init(struct ompi_communicator_t *comm, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, res, rsize;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  rsize = ompi_comm_remote_size (comm);

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  if (0 == rank) {
    for (int peer = 1 ; peer < rsize ; ++peer) {
      res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  }

  /* synchronize with the remote root */
  res = NBC_Sched_recv (NULL, false, 0, MPI_BYTE, 0, schedule, false);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, 0, schedule, false);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  if (0 == rank) {
    /* wait for the remote root */
    res = NBC_Sched_barrier (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    /* inform remote peers that all local peers have entered the barrier */
    for (int peer = 1; peer < rsize ; ++peer) {
      res = NBC_Sched_send (NULL, false, 0, MPI_BYTE, peer, schedule, false);
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
  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_ibarrier_inter(struct ompi_communicator_t *comm, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_barrier_inter_init(comm, request, module, false);
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

int ompi_coll_bullnbc_barrier_init(struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_barrier_init(comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_barrier_inter_init(struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_barrier_inter_init(comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
