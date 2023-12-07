/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
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
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "coll_bullnbc_internal.h"

static mca_base_var_enum_value_t scatterv_algorithms[] = {
    {0, "ignore"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_scatterv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != scatterv_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iscatterv_algorithm_count",
                                         "Number of scatterv algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_scatterv_algorithms", scatterv_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iscatterv_algorithm",
                                         "Which scatterv algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "iscatterv_algorithm_segmentsize",
                                  "Segment size in bytes used by default for iscatterv algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

/* a scatterv schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* simple linear MPI_Iscatterv */
static int nbc_scatterv_init(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, p, res;
  MPI_Aint sndext;
  BULLNBC_Schedule *schedule;
  char inplace = 0;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  if (root == rank) {
    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  }

  p = ompi_comm_size (comm);

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* receive from root */
  if (rank == root) {
    res = ompi_datatype_type_extent (sendtype, &sndext);
    if (MPI_SUCCESS != res) {
      NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
      OBJ_RELEASE(schedule);
      return res;
    }

    for (int i = 0 ; i < p ; ++i) {
      char * sbuf;
      sbuf = (char *) sendbuf + displs[i] * sndext;
      if (i == root) {
        if (!inplace) {
          /* if I am the root - just copy the message */
          res = NBC_Sched_copy (sbuf, false, sendcounts[i], sendtype,
                                recvbuf, false, recvcount, recvtype, schedule, false);
        } else {
          res = OMPI_SUCCESS;
        }
      } else {
        /* root sends the right buffer to the right receiver */
        res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false);
      }

      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    }
  } else {
    /* recv msg from root */
    res = NBC_Sched_recv (recvbuf, false, recvcount, recvtype, root, schedule, false);
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

int ompi_coll_bullnbc_iscatterv(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                               void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatterv_init(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
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

static int nbc_scatterv_inter_init (const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                    void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                    struct ompi_communicator_t *comm, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
    int res, rsize;
    MPI_Aint sndext;
    BULLNBC_Schedule *schedule;
    char *sbuf;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    rsize = ompi_comm_remote_size (comm);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* receive from root */
    if (MPI_ROOT != root && MPI_PROC_NULL != root) {
        /* recv msg from root */
        res = NBC_Sched_recv(recvbuf, false, recvcount, recvtype, root, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    } else if (MPI_ROOT == root) {
        res = ompi_datatype_type_extent(sendtype, &sndext);
        if (MPI_SUCCESS != res) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
            OBJ_RELEASE(schedule);
            return res;
        }

        for (int i = 0 ; i < rsize ; ++i) {
            sbuf = (char *)sendbuf + displs[i] * sndext;
            /* root sends the right buffer to the right receiver */
            res = NBC_Sched_send (sbuf, false, sendcounts[i], sendtype, i, schedule, false);
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

int ompi_coll_bullnbc_iscatterv_inter(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                     struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatterv_inter_init(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
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

int ompi_coll_bullnbc_scatterv_init(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                   void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                   struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatterv_init(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
                                comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_scatterv_inter_init(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                         void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                         struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatterv_inter_init(sendbuf, sendcounts, displs, sendtype, recvbuf, recvcount, recvtype, root,
                                      comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
