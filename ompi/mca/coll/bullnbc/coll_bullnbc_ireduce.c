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

#include "ompi_config.h"
#include "opal/align.h"
#include "opal/util/bit_ops.h"
#include "ompi/op/op.h"
#include "opal/util/show_help.h"
#include "opal/runtime/opal_params.h"

#include "coll_bullnbc_internal.h"
#include "coll_bullnbc_partitioned.h"

static inline int red_sched_binomial (int rank, int p, int root, const void *sendbuf, void *redbuf, char tmpredbuf, int count, MPI_Datatype datatype,
                                      MPI_Op op, char inplace, BULLNBC_Schedule *schedule, void *tmpbuf);
static inline int red_sched_chain(int rank, int p, int root, const void *sendbuf, void *recvbuf,
                                  int count, MPI_Datatype datatype, MPI_Op op, MPI_Aint ext,
                                  size_t size, BULLNBC_Schedule *schedule, size_t fragsize);

static inline int red_sched_linear (int rank, int rsize, int root, const void *sendbuf, void *recvbuf, void *tmpbuf, int count, MPI_Datatype datatype,
                                    MPI_Op op, BULLNBC_Schedule *schedule);
static inline int red_sched_redscat_gather(
    int rank, int comm_size, int root, const void *sbuf, void *rbuf,
    char tmpredbuf, int count, MPI_Datatype datatype, MPI_Op op, char inplace,
    BULLNBC_Schedule *schedule, void *tmp_buf, struct ompi_communicator_t *comm);

static mca_base_var_enum_value_t reduce_algorithms[] = {
    {0, "ignore"},
    {1, "binomial"},
    {2, "chain"},
    {3, "rabenseifner"},
    {0, NULL}
};

static mca_base_var_enum_value_t preducer_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binary"},
    {3, "binonial"},
    {4, "rabenseifner"},
    {5, "linear_parts"},
    {6, "binary_parts"},
    {7, "binonial_parts"},
    {0, NULL}
};
static int preducer_algorithm;
static int need_reproductibility = 0;

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_reduce_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != reduce_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ireduce_algorithm_count",
                                         "Number of reduce algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_reduce_algorithms", reduce_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ireduce_algorithm",
                                         "Which reduce algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 8192;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ireduce_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ireduce algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  preducer_algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_preducer_algorithms",
                                  preducer_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "preducer_algorithm",
                                         "Which preducer algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0,
                                         MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &preducer_algorithm);
  OBJ_RELEASE(new_enum);

  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "reproductible_preduce",
                                  "Ask for partitioned reduction collecitve to be"
                                  " reproductibles: same input, same number of"
                                  " rank means same output",
                                  MCA_BASE_VAR_TYPE_BOOL, NULL, 0,
                                  MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &need_reproductibility);
  return OMPI_SUCCESS;
}


enum reduce_algorithm { NBC_RED_BINOMIAL, NBC_RED_CHAIN, NBC_RED_REDSCAT_GATHER};

static enum reduce_algorithm
nbc_reduce_select_algorithm(size_t size, int count, char is_op_commutative, size_t *segsize,
                            struct ompi_communicator_t *comm,
                            const ompi_coll_bullnbc_module_t *bullnbc_module)
{
    enum reduce_algorithm alg;

    /* if op is not commutative we have to deal with it */
    if(!mca_coll_bullnbc_component.reduce_allow_non_commutative_support
       && !is_op_commutative) {
        *segsize = size * count; /* unused when algorithm is binomial but we should avoid debug outputs with uninitialized variables */
        return NBC_RED_BINOMIAL;
    }

    if (mca_coll_bullnbc_component.use_dynamic_rules
        && 0 != mca_coll_bullnbc_component.forced_params[REDUCE].algorithm) {
        *segsize = mca_coll_bullnbc_component.forced_params[REDUCE].segsize;
        /* -1 is to shift from algorithm ID to enum */
        alg = mca_coll_bullnbc_component.forced_params[REDUCE].algorithm - 1;
        return alg;
    }

    if(bullnbc_module->com_rules[REDUCE]) {
        int algorithm;
        int dummy1;
        int dummy2;
        int tmp_segsize;
        algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[REDUCE],
                                                             size * count, &dummy1, &tmp_segsize, &dummy2);
        if(algorithm) {
            *segsize = tmp_segsize;
            return algorithm - 1;/* -1 is to shift from algorithm ID to enum */
        }
    }

    /* algorithm selection */
    int p = ompi_comm_size (comm);
    int nprocs_pof2 = opal_next_poweroftwo(p) >> 1;
    if (is_op_commutative  && p > 2 && count >= nprocs_pof2) {
        alg = NBC_RED_REDSCAT_GATHER;
    } else if (p > 4 || size * count < 65536 || !is_op_commutative) {
        alg = NBC_RED_BINOMIAL;
    } else {
        alg = NBC_RED_CHAIN;
    }

    if (alg == NBC_RED_REDSCAT_GATHER &&
        !(is_op_commutative && p > 2 && count >= nprocs_pof2)) {
        alg = NBC_RED_CHAIN;
    }

    return alg;
}

static inline void
nbc_reduce_correct_segsize(const enum reduce_algorithm alg, size_t *segsize_ptr,
                           const int count, const size_t size)
{
    if (NBC_RED_CHAIN == alg) {
        *segsize_ptr = 8192;
    }
    if (0 == *segsize_ptr) {
        *segsize_ptr = count * size; /* only one frag */
    }
}

/* the non-blocking reduce */
static int
nbc_reduce_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
    int rank;
    int p;
    int res;
    size_t segsize = 0;
    size_t size;
    MPI_Aint ext;
    BULLNBC_Schedule *schedule;
    char *redbuf = NULL, inplace;
    void *tmpbuf = NULL;
    char tmpredbuf = 0;
    enum reduce_algorithm alg;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t *) module;
    ptrdiff_t span, gap;

    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
    rank = ompi_comm_rank(comm);
    p = ompi_comm_size(comm);

    res = ompi_datatype_type_extent(datatype, &ext);
    if (MPI_SUCCESS != res) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
        return res;
    }

    res = ompi_datatype_type_size(datatype, &size);
    if (MPI_SUCCESS != res) {
        NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
        return res;
    }

    /* only one node -> copy data */
    if (1 == p && (!persistent || inplace)) {
        if (!inplace) {
            res = NBC_Copy(sendbuf, count, datatype, recvbuf, count, datatype, comm);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }
        return bullnbc_get_noop_request(persistent, request);
    }

    if (0 == size || 0 == count) {
        return bullnbc_get_noop_request(persistent, request);
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (p == 1) {
        /* Do not read MCA configuration in trivial cases */
        res = NBC_Sched_copy((void *) sendbuf, false, count, datatype, recvbuf, false, count,
                             datatype, schedule, false);
        goto schedule_prepared;
    }

    span = opal_datatype_span(&datatype->super, count, &gap);
    alg = nbc_reduce_select_algorithm(size, count, ompi_op_is_commute(op), &segsize, comm,
                                      bullnbc_module);
    nbc_reduce_correct_segsize(alg, &segsize, count, size);
    opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                        "Bullnbc ireduce : algorithm %d segmentsize %zu", alg + 1, segsize);

    /* allocate temporary buffers */
    size_t tmpbuf_size = span;

    uint64_t flags;
    int dev_id;
    char is_sendbuf_cuda = 0;
    char is_recvbuf_cuda = 0;
    char *true_recvbuf = NULL;
    const char *true_sendbuf = NULL;
    if (GPU_LIKELYHOOD(opal_ireduce_use_device_pointers)) {
        is_sendbuf_cuda = !(rank == root && inplace);
        is_recvbuf_cuda = (rank == root);
    } else if (rank == root) {
        is_recvbuf_cuda = opal_accelerator.check_addr(recvbuf, &dev_id, &flags);
        is_sendbuf_cuda = !inplace && opal_accelerator.check_addr(sendbuf, &dev_id, &flags);
    } else {
        is_recvbuf_cuda = 0;
        is_sendbuf_cuda = opal_accelerator.check_addr(sendbuf, &dev_id, &flags);
    }

    if (GPU_LIKELYHOOD(is_sendbuf_cuda || is_recvbuf_cuda)) {
        /* TODO allocate less memory if only one of both buffer is GPU */
        tmpbuf_size += 2 * span;
    }

    if ((alg == NBC_RED_REDSCAT_GATHER || alg == NBC_RED_BINOMIAL) &&
        (rank != root)) {
        /* recvbuf may not be valid on non-root nodes */
        ptrdiff_t span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
        tmpbuf_size += span_align;
        redbuf = (char *) span_align - gap;
        tmpredbuf = 1;
    }
    tmpbuf = malloc(tmpbuf_size);
    if (OPAL_UNLIKELY(NULL == tmpbuf)) {
        OBJ_RELEASE(schedule);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* Change send or/and receive buffer to ensure all reductions to be made
     * on CPU buffers */

    if (GPU_LIKELYHOOD(is_recvbuf_cuda)) {
        /* Back up and replace receive buffer to be a valid CPU buffer.
         * Recvbuf is used for reductions */
        true_recvbuf = recvbuf;
        recvbuf = (char *) tmpbuf + tmpbuf_size - span;
    }

    /* Schedule sendbuf copy (of recvbuf if inplace) if it is on device. */
    if (GPU_LIKELYHOOD(is_sendbuf_cuda)) {
        true_sendbuf = sendbuf;
        sendbuf = (char *) tmpbuf + tmpbuf_size - 2 * span;
        res = NBC_Sched_copy((void *) true_sendbuf, false, count, datatype,
                             (void *) sendbuf, false, count, datatype,
                             schedule, true);
    } else if (GPU_LIKELYHOOD(rank == root && inplace && is_recvbuf_cuda)) {
        sendbuf = (char *) tmpbuf + tmpbuf_size - 2 * span;
        res = NBC_Sched_copy((void *) true_recvbuf, false, count, datatype,
                             (void *) sendbuf, false, count, datatype,
                             schedule, true);
    }

    if (rank == root) {
        /* root reduces in receive buffer */
        redbuf = recvbuf;
    }

    switch (alg) {
    case NBC_RED_BINOMIAL:
        res = red_sched_binomial(rank, p, root, sendbuf, redbuf, tmpredbuf, count, datatype, op,
                                 inplace, schedule, tmpbuf);
        break;
    case NBC_RED_CHAIN:
        res = red_sched_chain(rank, p, root, sendbuf, redbuf, count, datatype, op, ext, size,
                              schedule, segsize);
        break;
    case NBC_RED_REDSCAT_GATHER:
        res = red_sched_redscat_gather(rank, p, root, sendbuf, redbuf, tmpredbuf, count, datatype,
                                       op, inplace, schedule, tmpbuf, comm);
        break;
    }

    /* Receive buffer is on device, reduction have been made in a temporary CPU buffer,
     * Schedule a final copy to the actual user buffer */
    if (GPU_LIKELYHOOD(rank == root && is_recvbuf_cuda)) {
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            free(tmpbuf);
            return res;
        }

        res = NBC_Sched_copy((void *) redbuf, tmpredbuf, count, datatype,
                             true_recvbuf, false, count, datatype, schedule, true);
    }

schedule_prepared:
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

int ompi_coll_bullnbc_ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                             MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_init(sendbuf, recvbuf, count, datatype, op, root,
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

static int nbc_reduce_inter_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                 MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, res, rsize;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
  ptrdiff_t span, gap;
  void *tmpbuf;

  rank = ompi_comm_rank (comm);
  rsize = ompi_comm_remote_size (comm);

  span = opal_datatype_span(&datatype->super, count, &gap);
  tmpbuf = malloc (span);
  if (OPAL_UNLIKELY(NULL == tmpbuf)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  res = red_sched_linear (rank, rsize, root, sendbuf, recvbuf, (void *)(-gap), count, datatype, op, schedule);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    OBJ_RELEASE(schedule);
    free(tmpbuf);
    return OMPI_ERR_OUT_OF_RESOURCE;
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
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_ireduce_inter(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                   MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_inter_init(sendbuf, recvbuf, count, datatype, op, root,
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


/* binomial reduce
 * if op is not commutative, reduce on rank 0, and then send the result to root rank
 *
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
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
#define RANK2VRANK(rank, vrank, root) \
{ \
  vrank = rank; \
  if (rank == 0) vrank = root; \
  if (rank == root) vrank = 0; \
}
#define VRANK2RANK(rank, vrank, root) \
{ \
  rank = vrank; \
  if (vrank == 0) rank = root; \
  if (vrank == root) rank = 0; \
}
static inline int red_sched_binomial (int rank, int p, int root, const void *sendbuf, void *redbuf, char tmpredbuf, int count, MPI_Datatype datatype,
                                      MPI_Op op, char inplace, BULLNBC_Schedule *schedule, void *tmpbuf) {
  int vroot, vrank, vpeer, peer, res, maxr;
  char *rbuf, *lbuf, *buf;
  int tmprbuf, tmplbuf;
  ptrdiff_t gap;
  (void)opal_datatype_span(&datatype->super, count, &gap);
  res = OMPI_SUCCESS;

  if (ompi_op_is_commute(op)) {
    vroot = root;
  } else {
    vroot = 0;
  }
  RANK2VRANK(rank, vrank, vroot);
  maxr = ceil_of_log2(p);

  if (rank != root) {
    inplace = 0;
  }

  /* ensure the result ends up in redbuf on vrank 0 */
  if (0 == (maxr%2)) {
    rbuf = (void *)(-gap);
    tmprbuf = true;
    lbuf = redbuf;
    tmplbuf = tmpredbuf;
  } else {
    lbuf = (void *)(-gap);
    tmplbuf = true;
    rbuf = redbuf;
    tmprbuf = tmpredbuf;
    if (inplace) {
        res = NBC_Sched_copy(rbuf, false, count, datatype,
                             ((char *)tmpbuf)-gap, false, count, datatype,
                             schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
    }
  }

  for (int r = 1, firstred = 1 ; r <= maxr ; ++r) {
    if ((vrank % (1 << r)) == 0) {
      /* we have to receive this round */
      vpeer = vrank + (1 << (r - 1));
      VRANK2RANK(peer, vpeer, vroot)
      if (peer < p) {
        int tbuf;
        /* we have to wait until we have the data */
        res = NBC_Sched_recv (rbuf, tmprbuf, count, datatype, peer, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }

        /* perform the reduce in my local buffer */
        /* this cannot be done until tmpbuf is unused :-( so barrier after the op */
        if (firstred && !inplace) {
          /* perform the reduce with the senbuf */
          res = NBC_Sched_op (sendbuf, false, rbuf, tmprbuf, count, datatype, op, schedule, true);
          firstred = 0;
        } else {
          /* perform the reduce in my local buffer */
          res = NBC_Sched_op (lbuf, tmplbuf, rbuf, tmprbuf, count, datatype, op, schedule, true);
        }

        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
        /* swap left and right buffers */
        buf = rbuf; rbuf = lbuf ; lbuf = buf;
        tbuf = tmprbuf; tmprbuf = tmplbuf; tmplbuf = tbuf;
      }
    } else {
      /* we have to send this round */
      vpeer = vrank - (1 << (r - 1));
      VRANK2RANK(peer, vpeer, vroot)
      if (firstred && !inplace) {
        /* we have to use the sendbuf in the first round .. */
        res = NBC_Sched_send (sendbuf, false, count, datatype, peer, schedule, false);
      } else {
        /* and the redbuf in all remaining rounds */
        res = NBC_Sched_send (lbuf, tmplbuf, count, datatype, peer, schedule, false);
      }

      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }

      /* leave the game */
      break;
    }
  }
  /* send to root if vroot ! root */
  if (vroot != root) {
    if (0 == rank) {
      res = NBC_Sched_send (redbuf, tmpredbuf, count, datatype, root, schedule, false);
    } else if (root == rank) {
      res = NBC_Sched_recv (redbuf, tmpredbuf, count, datatype, vroot, schedule, false);
    }
  }

  return res;
}

/* chain send ... */
static inline int
red_sched_chain(int rank, int p, int root, const void *sendbuf, void *recvbuf,
                int count, MPI_Datatype datatype, MPI_Op op, MPI_Aint ext,
                size_t size, BULLNBC_Schedule *schedule, size_t fragsize)
{
    int res, vrank, rpeer, speer, numfrag, fragcount;

    RANK2VRANK(rank, vrank, root);
    VRANK2RANK(rpeer, vrank + 1, root);
    VRANK2RANK(speer, vrank - 1, root);

    if (0 == count) {
        return OMPI_SUCCESS;
    }

    numfrag = count * size / fragsize;
    if ((count * size) % fragsize != 0) {
        numfrag++;
    }

    fragcount = count / numfrag;

    for (int fragnum = 0; fragnum < numfrag; ++fragnum) {
        long offset = fragnum * fragcount * ext;
        int thiscount = fragcount;
        if (fragnum == numfrag - 1) {
            /* last fragment may not be full */
            thiscount = count - fragcount * fragnum;
        }

        /* last node does not recv */
        if (vrank != p - 1) {
            if (vrank == 0 && sendbuf != recvbuf) {
                res = NBC_Sched_recv((char *) recvbuf + offset, false, thiscount, datatype, rpeer,
                                     schedule, true);
            } else {
                res = NBC_Sched_recv((char *) offset, true, thiscount, datatype, rpeer, schedule,
                                     true);
            }
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }

            /* root reduces into receivebuf */
            if (vrank == 0) {
                if (sendbuf != recvbuf) {
                    res = NBC_Sched_op((char *) sendbuf + offset, false, (char *) recvbuf + offset,
                                       false, thiscount, datatype, op, schedule, true);
                } else {
                    res = NBC_Sched_op((char *) offset, true, (char *) recvbuf + offset, false,
                                       thiscount, datatype, op, schedule, true);
                }
            } else {
                res = NBC_Sched_op((char *) sendbuf + offset, false, (char *) offset, true,
                                   thiscount, datatype, op, schedule, true);
            }

            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }

        /* root does not send */
        if (vrank != 0) {
            /* rank p-1 has to send out of sendbuffer :) */
            /* the barrier here seems awkward but isn't!!!! */
            if (vrank == p - 1) {
                res = NBC_Sched_send((char *) sendbuf + offset, false, thiscount, datatype, speer,
                                     schedule, true);
            } else {
                res = NBC_Sched_send((char *) offset, true, thiscount, datatype, speer, schedule,
                                     true);
            }

            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }
    }

    return OMPI_SUCCESS;
}

/* simple linear algorithm for intercommunicators */
static inline int red_sched_linear (int rank, int rsize, int root, const void *sendbuf, void *recvbuf, void *tmpbuf, int count, MPI_Datatype datatype,
                                    MPI_Op op, BULLNBC_Schedule *schedule) {
  int res;
  char *rbuf, *lbuf, *buf;

  if (0 == count) {
    return OMPI_SUCCESS;
  }

  if (MPI_ROOT == root) {
    int tmprbuf, tmplbuf;
    /* ensure the result ends up in recvbuf */
    if (0 == (rsize%2)) {
      lbuf = tmpbuf;
      tmplbuf = true;
      rbuf = recvbuf;
      tmprbuf = false;
    } else {
      rbuf = tmpbuf;
      tmprbuf = true;
      lbuf = recvbuf;
      tmplbuf = false;
    }

    res = NBC_Sched_recv (lbuf, tmplbuf, count, datatype, 0, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }

    for (int peer = 1 ; peer < rsize ; ++peer) {
      res = NBC_Sched_recv (rbuf, tmprbuf, count, datatype, peer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }

      res = NBC_Sched_op (lbuf, tmplbuf, rbuf, tmprbuf, count, datatype, op, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
      /* swap left and right buffers */
      buf = rbuf; rbuf = lbuf ; lbuf = buf;
      tmprbuf ^= 1; tmplbuf ^= 1;
    }
  } else if (MPI_PROC_NULL != root) {
    res = NBC_Sched_send (sendbuf, false, count, datatype, root, schedule, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

/*
 * red_sched_redscat_gather:
 *
 * Description: an implementation of Rabenseifner's Reduce algorithm [1, 2].
 *   [1] Rajeev Thakur, Rolf Rabenseifner and William Gropp.
 *       Optimization of Collective Communication Operations in MPICH //
 *       The Int. Journal of High Performance Computing Applications. Vol 19,
 *       Issue 1, pp. 49--66.
 *   [2] http://www.hlrs.de/mpi/myreduce.html.
 *
 * This algorithm is a combination of a reduce-scatter implemented with
 * recursive vector halving and recursive distance doubling, followed either
 * by a binomial tree gather.
 *
 * Step 1. If the number of processes is not a power of two, reduce it to
 * the nearest lower power of two (p' = 2^{\floor{\log_2 p}})
 * by removing r = p - p' extra processes as follows. In the first 2r processes
 * (ranks 0 to 2r - 1), all the even ranks send the second half of the input
 * vector to their right neighbor (rank + 1), and all the odd ranks send
 * the first half of the input vector to their left neighbor (rank - 1).
 * The even ranks compute the reduction on the first half of the vector and
 * the odd ranks compute the reduction on the second half. The odd ranks then
 * send the result to their left neighbors (the even ranks). As a result,
 * the even ranks among the first 2r processes now contain the reduction with
 * the input vector on their right neighbors (the odd ranks). These odd ranks
 * do not participate in the rest of the algorithm, which leaves behind
 * a power-of-two number of processes. The first r even-ranked processes and
 * the last p - 2r processes are now renumbered from 0 to p' - 1.
 *
 * Step 2. The remaining processes now perform a reduce-scatter by using
 * recursive vector halving and recursive distance doubling. The even-ranked
 * processes send the second half of their buffer to rank + 1 and the odd-ranked
 * processes send the first half of their buffer to rank - 1. All processes
 * then compute the reduction between the local buffer and the received buffer.
 * In the next log_2(p') - 1 steps, the buffers are recursively halved, and the
 * distance is doubled. At the end, each of the p' processes has 1 / p' of the
 * total reduction result.
 *
 * Step 3. A binomial tree gather is performed by using recursive vector
 * doubling and distance halving. In the non-power-of-two case, if the root
 * happens to be one of those odd-ranked processes that would normally
 * be removed in the first step, then the role of this process and process 0
 * are interchanged.
 *
 * Limitations:
 *   count >= 2^{\floor{\log_2 p}}
 *   commutative operations only
 *   intra-communicators only
 *
 * Memory requirements (per process):
 *   rank != root: 2 * count * typesize + 4 * \log_2(p) * sizeof(int) = O(count)
 *   rank == root: count * typesize + 4 * \log_2(p) * sizeof(int) = O(count)
 *
 * Schedule length (rounds): O(\log(p))
 * Recommendations: root = 0, otherwise it is required additional steps
 *                  in the root process.
 */
static inline int red_sched_redscat_gather(
    int rank, int comm_size, int root, const void *sbuf, void *rbuf,
    char tmpredbuf, int count, MPI_Datatype datatype, MPI_Op op, char inplace,
    BULLNBC_Schedule *schedule, void *tmp_buf, struct ompi_communicator_t *comm)
{
    int res = OMPI_SUCCESS;
    int *rindex = NULL, *rcount = NULL, *sindex = NULL, *scount = NULL;

    /* Find nearest power-of-two less than or equal to comm_size */
    int nsteps = opal_hibit(comm_size, comm->c_cube_dim + 1);   /* ilog2(comm_size) */
    if (nsteps < 1) {
        /* This case never happens (for comm_size < 2 other algorithms are used) */
        return OMPI_ERR_NOT_SUPPORTED;
    }
    int nprocs_pof2 = 1 << nsteps;                              /* flp2(comm_size) */

    ptrdiff_t lb, extent;
    ompi_datatype_get_extent(datatype, &lb, &extent);

    if ((rank != root) || !inplace) {
        res = NBC_Sched_copy((char *)sbuf, false, count, datatype,
                             rbuf, tmpredbuf, count, datatype, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
    }

    /*
     * Step 1. Reduce the number of processes to the nearest lower power of two
     * p' = 2^{\floor{\log_2 p}} by removing r = p - p' processes.
     * 1. In the first 2r processes (ranks 0 to 2r - 1), all the even ranks send
     *    the second half of the input vector to their right neighbor (rank + 1)
     *    and all the odd ranks send the first half of the input vector to their
     *    left neighbor (rank - 1).
     * 2. All 2r processes compute the reduction on their half.
     * 3. The odd ranks then send the result to their left neighbors
     *    (the even ranks).
     *
     * The even ranks (0 to 2r - 1) now contain the reduction with the input
     * vector on their right neighbors (the odd ranks). The first r even
     * processes and the p - 2r last processes are renumbered from
     * 0 to 2^{\floor{\log_2 p}} - 1. These odd ranks do not participate in the
     * rest of the algorithm.
     */

    int vrank, step, wsize;
    int nprocs_rem = comm_size - nprocs_pof2;

    if (rank < 2 * nprocs_rem) {
        int count_lhalf = count / 2;
        int count_rhalf = count - count_lhalf;

        if (rank % 2 != 0) {
            /*
             * Odd process -- exchange with rank - 1
             * Send the left half of the input vector to the left neighbor,
             * Recv the right half of the input vector from the left neighbor
             */
            res = NBC_Sched_send(rbuf, tmpredbuf, count_lhalf, datatype, rank - 1,
                                 schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            res = NBC_Sched_recv((char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                                 false, count_rhalf, datatype, rank - 1, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            res = NBC_Sched_op((char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                               false, (char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                               tmpredbuf, count_rhalf, datatype, op, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Send the right half to the left neighbor */
            res = NBC_Sched_send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                 tmpredbuf, count_rhalf, datatype, rank - 1, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* This process does not participate in recursive doubling phase */
            vrank = -1;

        } else {
            /*
             * Even process -- exchange with rank + 1
             * Send the right half of the input vector to the right neighbor,
             * Recv the left half of the input vector from the right neighbor
             */
            res = NBC_Sched_send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                 tmpredbuf, count_rhalf, datatype, rank + 1, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            res = NBC_Sched_recv((char *)tmp_buf, false, count_lhalf, datatype, rank + 1,
                                 schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            res = NBC_Sched_op(tmp_buf, false, rbuf, tmpredbuf, count_lhalf,
                               datatype, op, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Recv the right half from the right neighbor */
            res = NBC_Sched_recv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                 tmpredbuf, count_rhalf, datatype, rank + 1, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            vrank = rank / 2;
        }
    } else { /* rank >= 2 * nprocs_rem */
        vrank = rank - nprocs_rem;
    }

    /*
     * Step 2. Reduce-scatter implemented with recursive vector halving and
     * recursive distance doubling. We have p' = 2^{\floor{\log_2 p}}
     * power-of-two number of processes with new ranks (vrank) and result in rbuf.
     *
     * The even-ranked processes send the right half of their buffer to rank + 1
     * and the odd-ranked processes send the left half of their buffer to
     * rank - 1. All processes then compute the reduction between the local
     * buffer and the received buffer. In the next \log_2(p') - 1 steps, the
     * buffers are recursively halved, and the distance is doubled. At the end,
     * each of the p' processes has 1 / p' of the total reduction result.
     */

    rindex = malloc(sizeof(*rindex) * nsteps);    /* O(\log_2(p)) */
    sindex = malloc(sizeof(*sindex) * nsteps);
    rcount = malloc(sizeof(*rcount) * nsteps);
    scount = malloc(sizeof(*scount) * nsteps);
    if (NULL == rindex || NULL == sindex || NULL == rcount || NULL == scount) {
        res = OMPI_ERR_OUT_OF_RESOURCE;
        goto cleanup_and_return;
    }

    if (vrank != -1) {
        step = 0;
        wsize = count;
        sindex[0] = rindex[0] = 0;

        for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
            /*
             * On each iteration: rindex[step] = sindex[step] -- begining of the
             * current window. Length of the current window is storded in wsize.
             */
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;

            if (rank < dest) {
                /*
                 * Recv into the left half of the current window, send the right
                 * half of the window to the peer (perform reduce on the left
                 * half of the current window)
                 */
                rcount[step] = wsize / 2;
                scount[step] = wsize - rcount[step];
                sindex[step] = rindex[step] + rcount[step];
            } else {
                /*
                 * Recv into the right half of the current window, send the left
                 * half of the window to the peer (perform reduce on the right
                 * half of the current window)
                 */
                scount[step] = wsize / 2;
                rcount[step] = wsize - scount[step];
                rindex[step] = sindex[step] + scount[step];
            }

            /* Send part of data from the rbuf, recv into the tmp_buf */
            res = NBC_Sched_send((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                 tmpredbuf, scount[step], datatype, dest, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            res = NBC_Sched_recv((char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                                 false, rcount[step], datatype, dest, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Local reduce: rbuf[] = tmp_buf[] <op> rbuf[] */
            res = NBC_Sched_op((char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                               false, (char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                               tmpredbuf, rcount[step], datatype, op, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Move the current window to the received message */
            if (step + 1 < nsteps) {
                rindex[step + 1] = rindex[step];
                sindex[step + 1] = rindex[step];
                wsize = rcount[step];
                step++;
            }
        }
    }
    /*
     * Assertion: each process has 1 / p' of the total reduction result:
     * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
     */

    /*
     * Setup the root process for gather operation.
     * Case 1: root < 2r and root is odd -- root process was excluded on step 1
     *         Recv data from process 0, vroot = 0, vrank = 0
     * Case 2: root < 2r and root is even: vroot = root / 2
     * Case 3: root >= 2r: vroot = root - r
     */
    int vroot = 0;
    if (root < 2 * nprocs_rem) {
        if (root % 2 != 0) {
            vroot = 0;
            if (rank == root) {
                /*
                 * Case 1: root < 2r and root is odd -- root process was
                 * excluded on step 1 (newrank == -1).
                 * Recv a data from the process 0.
                 */
                rindex[0] = 0;
                step = 0, wsize = count;
                for (int mask = 1; mask < nprocs_pof2; mask *= 2) {
                    rcount[step] = wsize / 2;
                    scount[step] = wsize - rcount[step];
                    rindex[step] = 0;
                    sindex[step] = rcount[step];
                    step++;
                    wsize /= 2;
                }

                res = NBC_Sched_recv(rbuf, tmpredbuf, rcount[nsteps - 1], datatype,
                                     0, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                vrank = 0;

            } else if (vrank == 0) {
                /* Send a data to the root */
                res = NBC_Sched_send(rbuf, tmpredbuf, rcount[nsteps - 1], datatype,
                                     root, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                vrank = -1;
            }
        } else {
            /* Case 2: root < 2r and a root is even: vroot = root / 2 */
            vroot = root / 2;
        }
    } else {
        /* Case 3: root >= 2r: newroot = root - r */
        vroot = root - nprocs_rem;
    }

    /*
     * Step 3. Gather result at the vroot by the binomial tree algorithm.
     * Each process has 1 / p' of the total reduction result:
     * rcount[nsteps - 1] elements in the rbuf[rindex[nsteps - 1], ...].
     * All exchanges are executed in reverse order relative
     * to recursive doubling (previous step).
     */

    if (vrank != -1) {
        int vdest_tree, vroot_tree;
        step = nsteps - 1; /* step = ilog2(p') - 1 */

        for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
            int vdest = vrank ^ mask;
            /* Translate vdest virtual rank to real rank */
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
            if ((vdest == 0) && (root < 2 * nprocs_rem) && (root % 2 != 0))
                dest = root;

            vdest_tree = vdest >> step;
            vdest_tree <<= step;
            vroot_tree = vroot >> step;
            vroot_tree <<= step;
            if (vdest_tree == vroot_tree) {
                /* Send data from rbuf and exit */

                res = NBC_Sched_send((char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                                     tmpredbuf, rcount[step], datatype, dest, schedule, false);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                break;
            } else {
                /* Recv and continue */
                res = NBC_Sched_recv((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                     tmpredbuf, scount[step], datatype, dest, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            }
            step--;
        }
    }

  cleanup_and_return:
    if (NULL != rindex)
        free(rindex);
    if (NULL != sindex)
        free(sindex);
    if (NULL != rcount)
        free(rcount);
    if (NULL != scount)
        free(scount);
    return res;
}

int ompi_coll_bullnbc_reduce_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                 MPI_Op op, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_init(sendbuf, recvbuf, count, datatype, op, root,
                              comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_reduce_inter_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                       MPI_Op op, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_reduce_inter_init(sendbuf, recvbuf, count, datatype, op, root,
                                    comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}





/* Schedule a reduce based on a tree, with root rank on top of the tree.
 * This routine just need the local info: the parent and the children
 */
static int
generic_partitioned_reduce_tree (const void *sbuf, void* rbuf,
                                 size_t parts, int count,
                                 struct ompi_datatype_t *dtype,
                                 ompi_request_t ** sendreqs,
                                 ompi_request_t ** recvreqs,
                                 struct ompi_op_t *op, int root,
                                 struct ompi_communicator_t *comm,
                                 ompi_request_t ** request,
                                 int parent,
                                 int nchild, int *child)
{

    int rank = ompi_comm_rank (comm);
    int nsend = rank == root ? 0 : 1;
    int nrecv = nchild;

    int ret;

    /* Allocations */
    ompi_coll_bullnbc_pcoll_request_t * req;
    int n_recvreq;
    if (rank != root) {
        if (OPAL_UNLIKELY(MPIX_NO_REQUESTS != recvreqs)) {
            NBC_Error("MPI Error in Preducer : recv reqs is not NULL");
            return OMPI_ERR_BAD_PARAM;
        }
        n_recvreq = 0;
    } else {
        n_recvreq = parts;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(parts, sendreqs,
                                               n_recvreq, recvreqs,
                                               nsend + nrecv, nrecv,
                                               comm, request);

    MPI_Aint extent;
    ret = ompi_datatype_type_extent(dtype, &extent);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }

    int tmpsize;
    if (0 == nchild) {
        tmpsize = 0;
    } else if (rank == root) {
        tmpsize = nrecv;
    } else {
        tmpsize = nrecv +1; /* +1 :reduction buffer */
    }
    req->tmpbuf = bullnbc_xmalloc(tmpsize * parts * count * extent);

    /* Temporary buffer to reduce and send data */
    char * reduction_buf;
    if (rank == root) {
        reduction_buf = rbuf;
    } else if (0 == nchild) {
        reduction_buf = (char*)sbuf;
    } else {
        reduction_buf = ((char*)req->tmpbuf) + nchild*count*parts*extent;
    }

    pcoll_dag_item** nodes = bullnbc_xmalloc(3*parts * sizeof(pcoll_dag_item*));
    for (size_t part=0; part < parts; ++part){
        nodes[part] = sched_ready(part, req, 0);
        nodes[parts + part] = sched_complete(part, req);
    }

    if (rank == root) {
        /* complete recvreqs */
        for (size_t part=0; part < parts; ++part){
            nodes[2*parts + part] = sched_complete(parts + part, req);
        }
        if (sbuf == MPI_IN_PLACE) {
            for (size_t part=0; part < parts; ++part){
                /* ready -> complete sendreq */
                create_dependency(nodes[part], nodes[parts + part]);
            }
        }
    }

    pcoll_dag_item* send = NULL;
    if (rank != root) {
        send = sched_send_v2(reduction_buf, count*parts, dtype,
                          parent, comm, req);
    }

    if (0 == nchild) {
        /* cannot be root: algo switch prevent binary reduce on a single proc comm */
        for (size_t part=0; part < parts; ++part){
            create_dependency(nodes[part], send); /* ready */
            create_dependency(send, nodes[parts + part]); /* reqs completion */
        }
    } else {

        /* Handle of MPI_IN_PLACE: copy */
        pcoll_dag_item * move = NULL;
        if (sbuf != MPI_IN_PLACE) {
            move = sched_convertion(sbuf, count*parts, dtype,
                                    reduction_buf, count*parts, dtype, 0);
            for (size_t part=0; part < parts; ++part){
                create_dependency(nodes[part], move); /* ready */
                create_dependency(move, nodes[parts + part]); /* reqs completion */
            }
            if (root != rank) {
                create_dependency(move, send);
            }
        }

        pcoll_dag_item* last_op = NULL;
        for (int i = 0; i < nchild; ++i) {
            char* child_rbuf = ((char*)req->tmpbuf) +i*count*parts*extent;
            pcoll_dag_item* recv;
            recv = sched_recv_v2(child_rbuf, count*parts, dtype,
                              child[i], comm, req);

            pcoll_dag_item* op_item;
            op_item = sched_op(reduction_buf, child_rbuf, parts*count,
                               dtype, op, 0);

            if (need_reproductibility && NULL != last_op){
                create_dependency(last_op, op_item);
                last_op = op_item;
            }
            create_dependency(recv, op_item);

            if (sbuf != MPI_IN_PLACE) {
                create_dependency(move, op_item);
            } else {
                for (size_t part=0; part < parts; ++part){
                    create_dependency(nodes[part], op_item); /*ready */
                }
            }

            if (root == rank) {
                for (size_t part=0; part < parts; ++part){
                    /* recvreqs completion */
                    create_dependency(op_item, nodes[2*parts + part]); 
                }
            } else {
                create_dependency(op_item, send);
            }
        }
    }

    free(nodes);

    return OMPI_SUCCESS;
}

static int
generic_partitioned_reduce_tree_per_parts (const void *sbuf, void* rbuf,
                                           size_t parts, int count,
                                           struct ompi_datatype_t *dtype,
                                           ompi_request_t ** sendreqs,
                                           ompi_request_t ** recvreqs,
                                           struct ompi_op_t *op, int root,
                                           struct ompi_communicator_t *comm,
                                           ompi_request_t ** request,
                                           int parent,
                                           int nchild, int *child)
{

    int rank = ompi_comm_rank (comm);
    int nsend = rank == root ? 0 : parts;
    int nrecv = nchild * parts;

    int ret;

    /* Allocations */
    ompi_coll_bullnbc_pcoll_request_t * req;
    int n_recvreq;
    if (rank != root) {
        if (OPAL_UNLIKELY(MPIX_NO_REQUESTS != recvreqs)) {
            NBC_Error("MPI Error in Preducer : recv reqs is not NULL");
            return OMPI_ERR_BAD_PARAM;
        }
        n_recvreq = 0;
    } else {
        n_recvreq = parts;
    }
    req = coll_bullnbc_pcoll_init_dag_ntag_req(parts, sendreqs,
                                               n_recvreq, recvreqs,
                                               nsend + nrecv, nrecv,
                                               comm, parts, request);

    MPI_Aint extent;
    ret = ompi_datatype_type_extent(dtype, &extent);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }

    int tmpsize;
    if (0 == nchild) {
        tmpsize = 0;
    } else if (rank == root) {
        tmpsize = nchild;
    } else {
        tmpsize = nchild +1; /* +1 :reduction buffer */
    }
    req->tmpbuf = bullnbc_xmalloc(tmpsize * parts * count * extent);

    /* Temporary buffer to reduce and send data */
    char * reduction_buf;
    if (rank == root) {
        reduction_buf = rbuf;
    } else if (0 == nchild) {
        reduction_buf = (char*)sbuf;
    } else {
        reduction_buf = ((char*)req->tmpbuf) + nchild*count*parts*extent;
    }

    for (size_t part=0; part < parts; ++part){

        pcoll_dag_item * ready, *scomplete, *rcomplete = NULL;
        ready = sched_ready(part, req, 0);
        scomplete = sched_complete(part, req);

        if (rank == root) {
            rcomplete = sched_complete(parts + part, req);
            if (sbuf == MPI_IN_PLACE) {
                create_dependency(ready, scomplete);
            }
        }

        char * part_reduction_buf = reduction_buf + part*count*extent;
        pcoll_dag_item* send = NULL;
        if (rank != root) {
            send = sched_send_tagged_v2(part_reduction_buf, count, dtype,
                                        parent, part, comm, req);
        }

        if (0 == nchild) {
            /* cannot be root: algo switch prevent binary reduce on a single proc comm */
            create_dependency(ready, send);
            create_dependency(send, scomplete);
        } else {

            /* Handle of MPI_IN_PLACE: copy */
            pcoll_dag_item * move = NULL;
            if (sbuf != MPI_IN_PLACE) {
                move = sched_convertion(((const char*)sbuf)+part*count*extent,
                                        count, dtype,
                                        part_reduction_buf,
                                        count, dtype,
                                        0);
                create_dependency(ready, move); /* ready */
                create_dependency(move, scomplete); /* reqs completion */
                if (root != rank) {
                    create_dependency(move, send);
                }
            }

            pcoll_dag_item* last_op = NULL;
            for (int i = 0; i < nchild; ++i) {
                char* child_rbuf = ((char*)req->tmpbuf) +(i*parts +part)*count*extent;
                pcoll_dag_item* recv;
                recv = sched_recv_tagged_v2(child_rbuf, count, dtype,
                                            child[i], part, comm, req);

                pcoll_dag_item* op_item;
                op_item = sched_op(part_reduction_buf, child_rbuf, count,
                                   dtype, op, 0);

                if (need_reproductibility && NULL != last_op){
                    create_dependency(last_op, op_item);
                    last_op = op_item;
                }
                create_dependency(recv, op_item);

                if (sbuf != MPI_IN_PLACE) {
                    create_dependency(move, op_item);
                } else {
                    /* Must be root, reduction operation targets rbuf */
                    create_dependency(ready, op_item);
                }

                if (root == rank) {
                    create_dependency(op_item, rcomplete);
                } else {
                    create_dependency(op_item, send);
                }
            }
        }
    }

    return OMPI_SUCCESS;
}


/*
 * Schedule a linear reduce, where all rank sched a send to root,
 * and root schedule a reduction after each recv */
static int
ompi_coll_bullnbc_preducer_linear (const void *sbuf, void* rbuf,
                                   size_t parts, int count,
                                   struct ompi_datatype_t *dtype,
                                   ompi_request_t ** sendreqs,
                                   ompi_request_t ** recvreqs,
                                   struct ompi_op_t *op, int root,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module,
                                   int part_pipelined)
{
    int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    int parent = root;
    const int nchild = rank == root ? comm_size-1 : 0;
    int child[nchild];
    for (int i=0; i < nchild; ++i){
        child[i] = (i < root) ? i : i +1;
    }
    if (part_pipelined) {
        return  generic_partitioned_reduce_tree_per_parts(sbuf, rbuf, parts, count, dtype,
                                                          sendreqs, recvreqs,
                                                          op, root, comm, request,
                                                          parent, nchild, child);
    } else {
        return  generic_partitioned_reduce_tree(sbuf, rbuf, parts, count, dtype,
                                                sendreqs, recvreqs,
                                                op, root, comm, request,
                                                parent, nchild, child);
    }
}

static int
ompi_coll_bullnbc_preducer_binary (const void *sbuf, void* rbuf,
                                   size_t parts, int count,
                                   struct ompi_datatype_t *dtype,
                                   ompi_request_t ** sendreqs,
                                   ompi_request_t ** recvreqs,
                                   struct ompi_op_t *op, int root,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module,
                                   int part_pipelined)
{
    int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    int vrank = (rank - root + comm_size) % comm_size;



    /* Build in order binary tree with 0 as root: reverse ranking order
     * compared to ompi_coll_base_topo_build_in_order_bintree.
     * The last send/recv is avoided at the end of reduce */
    int vparent = -1, my_pos = vrank, subtree_size = comm_size, delta = 0;
    int child[2], lchild, rchild;
    while (1) {
    /* Look for the subtree of the global binary tree, in which this
     * process is root.
     * In order binary tree splits ranks linearly:
     *  - Subtree roots are the smallest rank.
     *  - the N/2 lowest ranks are in the left child subtree,
     *  - the N/2 hihest ranks in the right child subtree.
     *  Therefore left child is always root+1 */
        rchild = -1;
        if (subtree_size > 2) {
            rchild = (subtree_size + 1)/2;
        }

        if (0 == my_pos) {
            if (subtree_size > 1) {
                lchild = delta + 1;
                if (subtree_size > 2) {
                    rchild += delta;
                }
            } else {
                lchild =-1;
            }
            break;
        }

        vparent = delta;

        if (subtree_size == 2 && my_pos == 1) { /* my_pos == 1 */
            subtree_size = 1;
            lchild = rchild = -1;
            break;
        }

        if (my_pos >= rchild) { /* Right part of the tree */
            delta += rchild;
            subtree_size -= rchild;
            my_pos -= rchild;
        } else {  /* child in left part of the tree */
            delta += 1;
            subtree_size = rchild -1;
            my_pos -= 1;
        }
        NBC_DEBUG(1, "parent=%d lchild=%d rchild=%d, subtree %d pos %d\n",
                vparent, vparent+1, rchild, subtree_size, my_pos);
    }

    int nchild = subtree_size > 2 ? 2: subtree_size -1;
    NBC_DEBUG(1, "vparent %d, %d children , left %d, right %d\n",
               vparent, nchild, lchild, rchild);
    child[0] = (lchild + root) % comm_size;
    child[1] = (rchild + root) % comm_size;
    int parent = (root + vparent) % comm_size;

    if (part_pipelined) {
        return  generic_partitioned_reduce_tree_per_parts(sbuf, rbuf, parts, count, dtype,
                                                          sendreqs, recvreqs,
                                                          op, root, comm, request,
                                                          parent, nchild, child);
    } else {
        return  generic_partitioned_reduce_tree(sbuf, rbuf, parts, count, dtype,
                                                sendreqs, recvreqs,
                                                op, root, comm, request,
                                                parent, nchild, child);
    }
}

static int
ompi_coll_bullnbc_preducer_binomial (const void *sbuf, void* rbuf,
                                     size_t parts, int count,
                                     struct ompi_datatype_t *dtype,
                                     ompi_request_t ** sendreqs,
                                     ompi_request_t ** recvreqs,
                                     struct ompi_op_t *op, int root,
                                     struct ompi_communicator_t *comm,
                                     struct ompi_info_t *info,
                                     ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module,
                                     int part_pipelined)
{
    int comm_size, rank, vrank;
    int nchild = 0;
    int parent = -1;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    vrank = (rank - root + comm_size) % comm_size;

    int child [100];
    while ((vrank + (1<<nchild) < comm_size) &&
           (!(vrank & (1<<nchild)))) {
        child[nchild] = (vrank + (1<<nchild) + root) % comm_size;
        nchild++;
    }
    if (vrank) {
        int vparent = 1<<nchild;
        while (!(vrank & vparent)) {
            vparent <<=1;
        }
        vparent = vrank - vparent;
        parent = (vparent + root) % comm_size;
    }

    if (part_pipelined) {
        return  generic_partitioned_reduce_tree_per_parts(sbuf, rbuf, parts, count, dtype,
                                                          sendreqs, recvreqs,
                                                          op, root, comm, request,
                                                          parent, nchild, child);
    } else {
        return  generic_partitioned_reduce_tree(sbuf, rbuf, parts, count, dtype,
                                                sendreqs, recvreqs,
                                                op, root, comm, request,
                                                parent, nchild, child);
    }
}


/* Schedule a reduction in three phases:
 * - a down sizing of the communicator sze to a power of two (the first ranks
 *   perform a first reduction with just another peer)
 * - a reduce scatter on the resized communicator in a recursive half doubling
 *   (at each iteration a half of the data is exchanged with peer rank+2^k)
 * - a gather to root of all the fragments, that happen to use the same
 *   peers as the reduce_scatter phase.
 *  At each step processes work by pairs. The smallest rank works on the
 *  first half of the data and send the remaining second half.
 *  The highest rank does the opposite */
static int
ompi_coll_bullnbc_preducer_rabenseifner (const void *sbuf, void* rbuf,
                                         size_t parts, int count,
                                         struct ompi_datatype_t *dtype,
                                         ompi_request_t ** sendreqs,
                                         ompi_request_t ** recvreqs,
                                         struct ompi_op_t *op, int root,
                                         struct ompi_communicator_t *comm,
                                         struct ompi_info_t *info,
                                         ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module)
{
    int comm_size, rank, vrank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);


    /* Find nearest power-of-two less than or equal to comm_size */
    int nsteps = opal_hibit(comm_size, comm->c_cube_dim + 1);
    if (OPAL_UNLIKELY(nsteps < 0)) {
        NBC_DEBUG(1, "%s called with commsize %d: surprisingly %d steps needed",
                  __func__, comm_size, nsteps);
        return OMPI_ERR_BAD_PARAM;
    }
    int log2_comm_size = 1<<nsteps;
    int rest_comm_size = comm_size - log2_comm_size;

    NBC_DEBUG(1, "expect %d iter, to cover %d+%d prcess\n",
              nsteps, log2_comm_size, rest_comm_size);

    MPI_Aint extent;
    int ret;
    ret = ompi_datatype_type_extent(dtype, &extent);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }


    /* Compute temporary buffer needs in terms of counts */
    int total_count = parts*count;
    int round_count = 0;
    /* A each step of the redscat I may have 1 count more than my remote*/
    int red_scat_count = total_count + nsteps;
    int gather_tmp_count = 0;

    int nrecv = 0, nsend = 0;
    vrank = (rank - root + comm_size) % comm_size;
    if (vrank < 2*rest_comm_size) {
        round_count = total_count;
        /* Work shared in peers to step down comm_size to a power of 2 */
        if (vrank % 2) {
            nrecv = 1;
            nsend = 2;
            vrank = -1;
        } else {
            nrecv = 2;
            nsend = 1;
            vrank /= 2;
        }
        NBC_DEBUG(1, "round down %d send, %d recv\n", nsend, nrecv);
    } else {
        vrank -= rest_comm_size;
    }

    if (vrank >= 0){
        int rank_hibit = opal_hibit(vrank, comm->c_cube_dim + 1);

        /* reduce_scatter nsteps, a sendrecv each*/
        nrecv += nsteps;
        nsend += nsteps;
        NBC_DEBUG(1, "redsct %d send, %d recv\n", nsend, nrecv);
        /* gather */
        nrecv += nsteps - (rank_hibit +1);
        nsend += (root == rank ? 0 : 1);
        NBC_DEBUG(1, "redsct+gather %d send, %d recv\n", nsend, nrecv);

        if (root != rank) {
            if (OPAL_UNLIKELY(rank_hibit < 0)) {
                /* 'rank != root' means 'vrank > 0' so hibit>=0
                 * but sonarqube finds a way with hibit<0, this test captures it */
                NBC_DEBUG(1, "%s called with comm_dimm %d by vrank %d(root=%d)"
                          " surprisingly hibit=%d<0",
                          __func__, comm->c_cube_dim, vrank, root, rank_hibit);
                return OMPI_ERR_BAD_PARAM;
            }
            gather_tmp_count = total_count / (1<<rank_hibit) / 2;
            gather_tmp_count += rest_comm_size; /* In case all frangments are not equals*/
        } else if (sbuf == MPI_IN_PLACE) {
            sbuf = rbuf;
        }
    }
    /* Reset vrank for the real scheduling */
    vrank = (rank - root + comm_size) % comm_size;


    /* Allocations */
    ompi_coll_bullnbc_pcoll_request_t * req;
    int ntag_need = 2; /* tag 0= before reduction, 1=after */
    int n_recvreq;
    if (rank != root) {
        if (OPAL_UNLIKELY(MPIX_NO_REQUESTS != recvreqs)) {
            NBC_Error("MPI Error in Preducer : recv reqs is not NULL");
            return ret;
        }
        n_recvreq = 0;
    } else {
        n_recvreq = parts;
    }
    req = coll_bullnbc_pcoll_init_dag_ntag_req(parts, sendreqs,
                                               n_recvreq, recvreqs,
                                               nsend + nrecv, nrecv,
                                               comm, ntag_need, request);

    pcoll_dag_item** nodes = bullnbc_xmalloc(3*parts * sizeof(pcoll_dag_item*));
    for (size_t part=0; part < parts; ++part){
        nodes[part] = sched_ready(part, req, 0);
        nodes[parts + part] = sched_complete(part, req);  //NOSONAR /* S3519 : Sonar thinks that nodes[parts + part] may be an invalid access */
    }
    if (rank == root) {
        for (size_t part=0; part < parts; ++part){
            nodes[2*parts + part] = sched_complete(req->total_sparts + part, req);
        }
    }

    int tmpcount = round_count; /* to round down process number to a power of 2 */
    /* TODO get N/commsize *2 smaller since last  step is done in gather buffer */
    tmpcount += red_scat_count; /* for receives in reduce_scatter */
    tmpcount += gather_tmp_count; /* to gather all reduced parts to root */

    NBC_DEBUG(1, "needs %d=%d+%d+%d extra memory\n",
            tmpcount, round_count, total_count, gather_tmp_count);
    req->tmpbuf = bullnbc_xmalloc(tmpcount * extent);

    char * round_buf = req->tmpbuf;
    char * scatter_rbuf = req->tmpbuf; /* where reduce_scatter recv data */
    char * scatter_buf = (void*)sbuf; /* where reduce_scatter read data */



    pcoll_dag_item* last_op = NULL;
    pcoll_dag_item* round_recv1 = NULL;
    pcoll_dag_item* round_recv2 = NULL;

    /***************************************
     * Round down comm size to a power of 2
     ***************************************/
    if (vrank <  2* rest_comm_size) {
        NBC_DEBUG(5, "Rabenseifner down sizing for %p\n", req);
        const void * round_sbuf;
        const void * round_kept_data;
        char* round_rbuf;
        char * round_remote_reduced;
        size_t round_sendcount, round_rcvcount;
        int peer;

        if (vrank % 2) { /* work on the second half of the data */
            round_sendcount = total_count /2 ;
            round_rcvcount = total_count - round_sendcount;

            round_kept_data = ((char*)sbuf) + round_sendcount * extent;
            round_sbuf = sbuf;
            round_rbuf = round_buf + round_sendcount * extent;
            peer = (vrank -1 +root) % comm_size;
        } else { /* work on the first half of the data */
            round_rcvcount = total_count /2;
            round_sendcount = total_count - round_rcvcount;

            round_kept_data = sbuf;
            round_sbuf = ((char*)sbuf) + round_rcvcount * extent;
            round_rbuf = round_buf;
            round_remote_reduced = round_buf + round_rcvcount * extent;
            peer = (vrank +1 +root) % comm_size;
        }
        /* Send a part of the data to make parallel computation */
        pcoll_dag_item* send;
        send = sched_send_tagged_v2(round_sbuf, round_sendcount, dtype,
                                 peer, 0, comm,
                                 req);
        for (size_t part = 0; part < parts; ++part) {
            create_dependency(nodes[part], send);
            create_dependency(send, nodes[parts + part]);
        }

        pcoll_dag_item* recv;
        recv = sched_recv_tagged_v2(round_rbuf, round_rcvcount, dtype,
                                 peer, 0, comm, req);
        round_recv1 = recv;

        pcoll_dag_item* op_item;
        op_item = sched_op(round_rbuf, round_kept_data , round_rcvcount,
                           dtype, op, 0);
        create_dependency(recv, op_item);
        for (size_t part = 0; part < parts; ++part) {
            create_dependency(nodes[part], op_item);
        }


        if (vrank % 2) {
            /* Send back reduced data */
            send = sched_send_tagged_v2(round_rbuf, round_rcvcount, dtype,
                                     peer, 1, comm, req);
            create_dependency(op_item, send);
            for (size_t part = 0; part < parts; ++part) {
                /* Complete sreqs */
                create_dependency(send, nodes[parts + part]);
            }
            goto exit_first_odd_ranks;
        } else {
            recv = sched_recv_tagged_v2(round_remote_reduced, round_sendcount, dtype,
                                     peer, 1,  comm, req);
            vrank /= 2;
            scatter_buf = round_buf;
            scatter_rbuf += total_count * extent;
            round_recv2 = recv;
            last_op = op_item;
        }
    } else {
        /* Do not need to contribute to comm_size down sizing */
        vrank -= rest_comm_size;
    }

    /***************************************
     * Reduce_scatter
     ***************************************/
    NBC_DEBUG(5, "Rabenseifner red_scatter for %p\n", req);

    int * steps_count = bullnbc_xmalloc((nsteps+1) * sizeof(int));
    steps_count[0] = total_count;

    /* Data transfers are bigger on first steps, make them on near ranks
     * in case ranks distances correspond to hardware distance */
    for (int step = 0; step < nsteps; ++step) {
        int mask = 1<<step;
        int vpeer = vrank ^ mask;
        int peer = vpeer < rest_comm_size ? vpeer*2 : vpeer + rest_comm_size;
        peer = (peer + root) % comm_size;

        int scount, rcount;
        char * scatter_sbuf, *scatter_kept;
        if (vrank < vpeer) { /* work on the first half of the data */
            rcount = steps_count[step] /2;
            scount = steps_count[step] - rcount;
            scatter_kept = scatter_buf;
            scatter_sbuf = scatter_buf + rcount * extent;
        } else { /* work on the second half of the data */
            scount = steps_count[step] /2;
            rcount = steps_count[step] - scount;
            scatter_sbuf = scatter_buf;
            scatter_kept = scatter_buf + scount * extent;
        }

        NBC_DEBUG(1, "redscat#%d (->%d(=%d)), send %p+%d, recv %p+%d reduced with %p+%d\n",
                  step, vpeer, peer, scatter_sbuf, scount,
                  scatter_rbuf, rcount,
                  scatter_kept, rcount);

        pcoll_dag_item* send;
        send = sched_send_tagged_v2(scatter_sbuf, scount, dtype,
                                 peer, 0, comm, req);
        if (NULL != last_op) create_dependency(last_op, send);

        if (0 == step) {
            if (NULL != round_recv1) create_dependency(round_recv1, send);
            if (NULL != round_recv2) create_dependency(round_recv2, send);
            for (size_t part = 0; part < parts; ++part) {
                create_dependency(nodes[part], send);
                create_dependency(send, nodes[parts + part]);
            }
        }

        pcoll_dag_item* recv;
        recv = sched_recv_tagged_v2(scatter_rbuf, rcount, dtype,
                                 peer, 0, comm,
                                 req);

        pcoll_dag_item* op_item;
        op_item = sched_op(scatter_rbuf, scatter_kept , rcount,
                           dtype, op, 0);
        create_dependency(recv, op_item);
        if (NULL != last_op) create_dependency(last_op, op_item);
        if (0 == step) {
            if (NULL != round_recv1) create_dependency(round_recv1, op_item);
            if (NULL != round_recv2) create_dependency(round_recv2, op_item);
            for (size_t part = 0; part < parts; ++part) {
                create_dependency(nodes[part], op_item);
                create_dependency(op_item, nodes[parts + part]);
            }
        }
        round_recv2 = round_recv1 = NULL;
        last_op = op_item;


        scatter_buf = scatter_rbuf;
        /* Use a new buffer for each recv to handle out of order reception */
        scatter_rbuf += rcount * extent;
        steps_count[step+1] = rcount;
    }

    char * gather_sbuf = scatter_buf;
    /* At this point a recv can only occur if the matching send has been done.
     * rbuf can be rused even in case of MPI_IN_PLACE */
    char * gather_buf = (root == rank)? rbuf: scatter_buf;
    if (root == rank) {
        pcoll_dag_item * move;
            int last_rcount = steps_count[nsteps];
            move = sched_convertion(scatter_buf , last_rcount, dtype,
                                    rbuf, last_rcount,  dtype,
                                    0);
            create_dependency(last_op, move);
            for (size_t part = 0; part < parts; ++part) {
                create_dependency(move, nodes[2*parts + part]);
            }
    }

    /******************************************
     * Gather all reduced fragments to the root
     ******************************************/
    NBC_DEBUG(5, "Rabenseifner gather for %p\n", req);
    pcoll_dag_item ** before_send = bullnbc_xmalloc ((1+nsteps) * sizeof(pcoll_dag_item*));
    int nbefore_send = 0;
    before_send[nbefore_send++] = last_op;


    for (int step = nsteps -1; step >= 0; --step) {
        int mask = 1 << step;
        int vpeer = vrank ^ mask;
        int peer = vpeer < rest_comm_size ? vpeer *2 : vpeer + rest_comm_size;
        peer = (peer + root) % comm_size;

        if (vpeer < vrank) {
            pcoll_dag_item* send;
            send = sched_send_tagged_v2(gather_sbuf, steps_count[step+1], dtype,
                                     peer, 1, comm, req);
            for (int i = 0; i < nbefore_send; ++i) {
                create_dependency(before_send[i], send);
            }
            for (size_t part = 0; part < parts; ++part) {
                create_dependency(send, nodes[parts + part]);
            }
            break;
        } else {
            pcoll_dag_item* recv;
            char * gather_rbuf = gather_buf + steps_count[step+1] * extent;
            int rcount = steps_count[step] - steps_count[step+1]; /* scatter scount */
            if (root == rank) {
                NBC_DEBUG(20, "RECV from %d at +%d\n",
                          peer, (ptrdiff_t)gather_rbuf - (ptrdiff_t)rbuf);
            }
            NBC_DEBUG(5, "gather recv(mask %d) of %d ddt from %d\n",
                      mask, rcount, peer);
            recv = sched_recv_tagged_v2(gather_rbuf, rcount, dtype,
                                     peer, 1, comm,
                                      req); /* rreq completion or send toward root*/
            before_send[nbefore_send++] = recv;
            if (root == rank) {
                for (size_t part = 0; part < parts; ++part) {
                    create_dependency(recv, nodes[2*parts + part]);
                }
            }
        }
    }

    free(before_send);
    free(steps_count);
exit_first_odd_ranks:
    free(nodes);
    return OMPI_SUCCESS;
}

int
ompi_coll_bullnbc_preduce_init (const void *sbuf, void* rbuf,
                                 size_t parts, int count,
                                 struct ompi_datatype_t *datatype,
                                 struct ompi_op_t *op, int root,
                                 struct ompi_communicator_t *comm,
                                 struct ompi_info_t *info,
                                 ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module)
{
    return ompi_coll_bullnbc_preducer_init(sbuf, rbuf, parts, count, datatype,
MPIX_NO_REQUESTS, MPIX_NO_REQUESTS, op, root, comm, info, request, module);
}


int
ompi_coll_bullnbc_preducer_init (const void *sbuf, void* rbuf,
                                 size_t parts, int count,
                                 struct ompi_datatype_t *datatype,
                                 ompi_request_t ** sendreqs,
                                 ompi_request_t ** recvreqs,
                                 struct ompi_op_t *op, int root,
                                 struct ompi_communicator_t *comm,
                                 struct ompi_info_t *info,
                                 ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module)
{
    size_t total_size;
    ompi_datatype_type_size(datatype, &total_size);
    total_size *= count*parts;

    int comm_size = ompi_comm_size (comm);
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);

    if ((sbuf == MPI_IN_PLACE && comm_size < 2)
        || 0 == datatype_size
        || 0 == count ) {
        ompi_mca_coll_bullnbc_pcoll_init_empty_req(parts, sendreqs,
                                                   parts, recvreqs,
                                                   comm, request);
        return MPI_SUCCESS;
    }

    int alg = preducer_algorithm;

    if (comm_size < 2) {
        alg = 1;    /* Just a copy to rbuf: done by linear */
    }

    if (0 == alg) {
        if (need_reproductibility) {
            alg = 4;
        } else {
            if (datatype_size * count * parts < 4096) {
                alg = 1;
            } else {
                alg = 2;
            }
        }
    }

    if (! mca_coll_bullnbc_uniform_partition_sizes) {
        int fallback_alg = alg;
        if (5 == alg) {
            fallback_alg = 1;
        }
        if (6 == alg) {
            fallback_alg = 2;
        }
        if (7 == alg) {
            fallback_alg = 3;
        }
        if (alg != fallback_alg) {
            opal_show_help("help-mpi-coll-bullnbc.txt",
                           "Non uniform partition sizes", true,
                           alg, "reduce", fallback_alg);
            alg = fallback_alg;
        }
    }

    module_need_progress((ompi_coll_bullnbc_module_t*) module);

    NBC_DEBUG(1, "Use algorithm %d for preducer with %d parts, count %d and root %d\n",alg, parts, count, root);
    switch (alg){
        case 1:
            return ompi_coll_bullnbc_preducer_linear(sbuf, rbuf, parts, count,
                                                     datatype, sendreqs, recvreqs,
                                                     op, root, comm, info,
                                                     request, module, 0);
        case 2:
            return ompi_coll_bullnbc_preducer_binary(sbuf, rbuf, parts, count,
                                                     datatype, sendreqs, recvreqs,
                                                     op, root, comm, info,
                                                     request, module, 0);
        case 3:
            return ompi_coll_bullnbc_preducer_binomial(sbuf, rbuf, parts, count,
                                                       datatype, sendreqs, recvreqs,
                                                       op, root, comm, info,
                                                       request, module, 0);
        case 4:
            return ompi_coll_bullnbc_preducer_rabenseifner(sbuf, rbuf, parts, count,
                                                           datatype, sendreqs, recvreqs,
                                                           op, root, comm, info,
                                                           request, module);
        case 5:
            return ompi_coll_bullnbc_preducer_linear(sbuf, rbuf, parts, count,
                                                    datatype, sendreqs, recvreqs,
                                                    op, root, comm, info,
                                                    request, module, 1);
        case 6:
            return ompi_coll_bullnbc_preducer_binary(sbuf, rbuf, parts, count,
                                                     datatype, sendreqs, recvreqs,
                                                     op, root, comm, info,
                                                     request, module, 1);
        case 7:
            return ompi_coll_bullnbc_preducer_binomial(sbuf, rbuf, parts, count,
                                                       datatype, sendreqs, recvreqs,
                                                       op, root, comm, info,
                                                       request, module, 1);
        default:
            fprintf(stderr, "Preducer algorithm %d is not implemented yet\n", alg); fflush(stderr);
            abort();
    }
}
