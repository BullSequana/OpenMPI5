/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2013      The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015      Mellanox Technologies. All rights reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "nbc_internal.h"

/* an gatherw schedule can not be cached easily because the contents
 * ot the recvcounts array may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */


static int nbc_gatherw_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                            void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                            int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, p, res;
  NBC_Schedule *schedule;
  char inplace = 0;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  if (root == rank) {
    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  }
  p = ompi_comm_size (comm);

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* send to root */
  if (rank != root) {
    /* send msg to root */
    res = NBC_Sched_send (sendbuf, false, sendcount, sendtype, root, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }
  } else {
    for (int i = 0 ; i < p ; ++i) {
      char *rbuf;
      rbuf = (char *) recvbuf + displs[i];
      if (i == root) {
        if (!inplace) {
          /* if I am the root - just copy the message */
          res = NBC_Sched_copy ((void *)sendbuf, false, sendcount, sendtype,
                                rbuf, false, recvcounts[i], recvtypes[i], schedule, false);
          if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
          }
        }
      } else {
        /* root receives message to the right buffer */
        res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtypes[i], i, schedule, false);
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

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_igatherw(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                              int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gatherw_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtypes, root,
                               comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

static int nbc_gatherw_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                   void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                   int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int res, rsize;
  NBC_Schedule *schedule;
  char *rbuf;
  ompi_coll_libnbc_module_t *libnbc_module = (ompi_coll_libnbc_module_t*) module;

  rsize = ompi_comm_remote_size (comm);

  schedule = OBJ_NEW(NBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* send to root */
  if (MPI_ROOT != root && MPI_PROC_NULL != root) {
    /* send msg to root */
    res = NBC_Sched_send (sendbuf, false, sendcount, sendtype, root, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      return res;
    }
  } else if (MPI_ROOT == root) {
    for (int i = 0 ; i < rsize ; ++i) {
      rbuf = (char *) recvbuf + displs[i];
      /* root receives message to the right buffer */
      res = NBC_Sched_recv (rbuf, false, recvcounts[i], recvtypes[i], i, schedule, false);
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

  res = NBC_Schedule_request(schedule, comm, libnbc_module, persistent, request, NULL);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    return res;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_libnbc_igatherw_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                    void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                    int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gatherw_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtypes, root,
                                     comm, request, module, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
  
    res = NBC_Start(*(ompi_coll_libnbc_request_t **)request);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        NBC_Return_handle (*(ompi_coll_libnbc_request_t **)request);
        *request = &ompi_request_null.request;
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_gatherw_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                  int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gatherw_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtypes, root,
                               comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_libnbc_gatherw_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                        void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                        int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gatherw_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtypes, root,
                                     comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
