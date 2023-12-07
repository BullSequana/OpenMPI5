/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2012      Sandia National Laboratories. All rights reserved.
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
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 */
#include "opal/align.h"
#include "opal/runtime/opal_params.h"

#include "coll_bullnbc_internal.h"

static mca_base_var_enum_value_t reduce_scatter_block_algorithms[] = {
    {0, "ignore"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_reduce_scatter_block_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != reduce_scatter_block_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ireduce_scatter_block_algorithm_count",
                                         "Number of reduce_scatter_block algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_reduce_scatter_block_algorithms", reduce_scatter_block_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ireduce_scatter_block_algorithm",
                                         "Which reduce_scatter_block algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ireduce_scatter_block_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ireduce_scatter_block algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

/* an reduce_csttare schedule can not be cached easily because the contents
 * ot the recvcount value may change, so a comparison of the address
 * would not be sufficient ... we simply do not cache it */

/* binomial reduce to rank 0 followed by a linear scatter ...
 *
 * Algorithm:
 * pairwise exchange
 * round r:
 *  grp = rank % 2^r
 *  if grp == 0: receive from rank + 2^(r-1) if it exists and reduce value
 *  if grp == 1: send to rank - 2^(r-1) and exit function
 *
 * do this for R=log_2(p) rounds
 *
 */

static int nbc_reduce_scatter_block_init( //NOSONAR cognitive complexity
                                         const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                         MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, maxr, p, res, count;
  MPI_Aint ext;
  ptrdiff_t gap;
  char inplace;
  BULLNBC_Schedule *schedule;
  void *tmpbuf = NULL;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  NBC_IN_PLACE(sendbuf, recvbuf, inplace);

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_extent(datatype, &ext);
  if (MPI_SUCCESS != res || 0 == ext) {
    NBC_Error ("MPI Error in ompi_datatype_type_extent() (%i:%i)", res, (int) ext);
    return (MPI_SUCCESS == res) ? MPI_ERR_SIZE : res;
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (NULL == schedule) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  maxr = ceil_of_log2(p);

  count = p * recvcount;

  if (0 < count) {
    char *rbuf, *lbuf, *buf, *redbuf;
    ptrdiff_t span_align;
    ptrdiff_t span;

    span = opal_datatype_span(&datatype->super, count, &gap);
    span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
    tmpbuf = malloc (span_align + span);
    if (NULL == tmpbuf) {
      OBJ_RELEASE(schedule);
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    rbuf = (void *)(-gap);
    lbuf = (char *)(span_align - gap);
    redbuf = (char *) tmpbuf + span_align - gap;

    /* copy data to redbuf if we only have a single node */
    if ((p == 1) && !inplace) {
      res = NBC_Sched_copy ((void *)sendbuf, false, count, datatype,
                            redbuf, false, count, datatype, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
    }

    /* Leaf are the only ranks that do not reduce data */
    char is_leaf = (p == rank+1 || rank % 2);
    int firstred = 1;
    uint64_t flags;
    int dev_id;
    if (!is_leaf && (opal_ireduce_scatter_block_use_device_pointers ||
                opal_accelerator.check_addr(sendbuf, &dev_id, &flags))) {
        /* First sched on sendbuf is OP, that needs a buffer on host */
        res = NBC_Sched_copy(sendbuf, false, count, datatype, lbuf, true, count, datatype, schedule,
                             true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
        firstred =0;
    }

    for (int r = 1 ; r <= maxr; ++r) {
      int peer;
      if ((rank % (1 << r)) == 0) {
        /* we have to receive this round */
        peer = rank + (1 << (r - 1));
        if (peer < p) {
          /* we have to wait until we have the data */
          res = NBC_Sched_recv (rbuf, true, count, datatype, peer, schedule, true);
          if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }

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
        if(firstred) {
          /* we have to send the senbuf */
          res = NBC_Sched_send (sendbuf, false, count, datatype, peer, schedule, false);
        } else {
          /* we send an already reduced value from redbuf */
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
    if (rank != 0) {
      res = NBC_Sched_recv (recvbuf, false, recvcount, datatype, 0, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
    } else {
      for (int r = 1, offset = 0 ; r < p ; ++r) {
        offset += recvcount;
        char *sbuf = lbuf + (offset*ext);
        /* root sends the right buffer to the right receiver */
        res = NBC_Sched_send (sbuf, true, recvcount, datatype, r, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
      }

      if ((p != 1) || !inplace) {
        res = NBC_Sched_copy (lbuf, true, recvcount, datatype, recvbuf, false, recvcount,
                              datatype, schedule, false);
      }
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_fail; }
    }
  }

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

int ompi_coll_bullnbc_ireduce_scatter_block(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                           struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_block_init(sendbuf, recvbuf, recvcount, datatype, op,
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

static int nbc_reduce_scatter_block_inter_init(const void *sendbuf, void *recvbuf, int rcount, struct ompi_datatype_t *dtype,
                                               struct ompi_op_t *op, struct ompi_communicator_t *comm, ompi_request_t **request,
                                               struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, res, count, lsize, rsize;
  MPI_Aint ext;
  ptrdiff_t gap, span, span_align;
  BULLNBC_Schedule *schedule;
  void *tmpbuf = NULL;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  lsize = ompi_comm_size (comm);
  rsize = ompi_comm_remote_size (comm);

  res = ompi_datatype_type_extent (dtype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error ("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  count = rcount * lsize;

  span = opal_datatype_span(&dtype->super, count, &gap);
  span_align = OPAL_ALIGN(span, dtype->super.align, ptrdiff_t);

  if (count > 0) {
    tmpbuf = malloc (span_align + span);
    if (NULL == tmpbuf) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (NULL == schedule) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  /* send my data to the remote root */
  res = NBC_Sched_send (sendbuf, false, count, dtype, 0, schedule, false);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return res;
  }

  if (0 == rank) {
    char *lbuf, *rbuf;
    lbuf = (char *)(-gap);
    rbuf = (char *)(span_align-gap);
    res = NBC_Sched_recv (lbuf, true, count, dtype, 0, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }

    for (int peer = 1 ; peer < rsize ; ++peer) {
      char *tbuf;
      res = NBC_Sched_recv (rbuf, true, count, dtype, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }

      res = NBC_Sched_op (lbuf, true, rbuf, true, count, dtype,
                          op, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
      tbuf = lbuf; lbuf = rbuf; rbuf = tbuf;
    }

    /* do the scatter with the local communicator */
    res = NBC_Sched_copy (lbuf, true, rcount, dtype, recvbuf, false, rcount,
                          dtype, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
    for (int peer = 1 ; peer < lsize ; ++peer) {
      res = NBC_Sched_local_send (lbuf + ext * rcount * peer, true, rcount, dtype, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        free(tmpbuf);
        return res;
      }
    }
  } else {
    /* receive my block */
    res = NBC_Sched_local_recv(recvbuf, false, rcount, dtype, 0, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      OBJ_RELEASE(schedule);
      free(tmpbuf);
      return res;
    }
  }

  /*NBC_PRINT_SCHED(*schedule);*/

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

int ompi_coll_bullnbc_ireduce_scatter_block_inter(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                                 MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                                 struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_block_inter_init(sendbuf, recvbuf, recvcount, datatype, op,
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

int ompi_coll_bullnbc_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                               MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_block_init(sendbuf, recvbuf, recvcount, datatype, op,
                                            comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_reduce_scatter_block_inter_init(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                                     MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                                     struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_scatter_block_inter_init(sendbuf, recvbuf, recvcount, datatype, op,
                                                  comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
