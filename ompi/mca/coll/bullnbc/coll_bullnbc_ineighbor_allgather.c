/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
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

static mca_base_var_enum_value_t neighbor_allgather_algorithms[] = {
    {0, "ignore"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_neighbor_allgather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != neighbor_allgather_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ineighbor_allgather_algorithm_count",
                                         "Number of neighbor_allgather algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_neighbor_allgather_algorithms", neighbor_allgather_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ineighbor_allgather_algorithm",
                                         "Which neighbor_allgather algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ineighbor_allgather_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ineighbor_allgather algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

static int nbc_neighbor_allgather_init(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                       int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                       ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int res, indegree, outdegree, *srcs, *dsts;
  MPI_Aint rcvext;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
  BULLNBC_Schedule *schedule;

  res = ompi_datatype_type_extent (rtype, &rcvext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    res = BULLNBC_Comm_neighbors (comm, &srcs, &indegree, &dsts, &outdegree);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }

    for (int i = 0 ; i < indegree ; ++i) {
      if (MPI_PROC_NULL != srcs[i]) {
        res = NBC_Sched_recv ((char *) rbuf + i * rcount * rcvext, true, rcount, rtype, srcs[i], schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          break;
        }
      }
    }

    free (srcs);

    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free (dsts);
      return res;
    }

    for (int i = 0 ; i < outdegree ; ++i) {
      if (MPI_PROC_NULL != dsts[i]) {
        res = NBC_Sched_send ((char *) sbuf, false, scount, stype, dsts[i], schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          break;
        }
      }
    }

    free (dsts);

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

int ompi_coll_bullnbc_ineighbor_allgather(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                         int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                         ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_neighbor_allgather_init(sbuf, scount, stype, rbuf, rcount, rtype,
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

int ompi_coll_bullnbc_neighbor_allgather_init(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                             int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                             MPI_Info info, ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_neighbor_allgather_init(sbuf, scount, stype, rbuf, rcount, rtype,
                                          comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
