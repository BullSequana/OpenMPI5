/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2014-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
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
#include "opal/util/show_help.h"

#include "coll_bullnbc_partitioned.h"

int bullnbc_ibcast_knomial_radix = 4;
static inline int bcast_sched_binomial(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count,
                                       MPI_Datatype datatype);
static inline int bcast_sched_linear(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count,
                                     MPI_Datatype datatype);
static inline int bcast_sched_chain(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count,
                                    MPI_Datatype datatype, int fragsize, size_t size);
static inline int bcast_sched_knomial(int rank, int comm_size, int root, BULLNBC_Schedule *schedule, void *buf,
                                      int count, MPI_Datatype datatype, int knomial_radix);

static mca_base_var_enum_value_t bcast_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binomial"},
    {3, "chain"},
    {4, "knomial"},
    {0, NULL}
};

static mca_base_var_enum_value_t pbcastr_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binomial"},
    {3, "chain"},
    {4, "knomial"},
    {5, "scatter_gather"},
    {6, "binomial_partitioned"},
#if OMPI_MPI_NOTIFICATIONS
#endif
    {0, NULL}
};
/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */
static int pbcastr_algorithm;


int ompi_coll_bullnbc_bcast_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != bcast_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ibcast_algorithm_count",
                                         "Number of bcast algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_bcast_algorithms", bcast_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "ibcast_algorithm",
                                         "Which bcast algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 16384;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "ibcast_algorithm_segmentsize",
                                  "Segment size in bytes used by default for ibcast algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
    bullnbc_ibcast_knomial_radix = 4;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "ibcast_knomial_radix", "k-nomial tree radix for the ibcast algorithm (radix > 1)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &bullnbc_ibcast_knomial_radix);

  OBJ_RELEASE(new_enum);


  pbcastr_algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_pbcastr_algorithms",
                                  pbcastr_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "pbcastr_algorithm",
                                         "Which pbcastr algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0,
                                         MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &pbcastr_algorithm);
  OBJ_RELEASE(new_enum);


  return OMPI_SUCCESS;
}

static int nbc_bcast_init(void *buffer, int count, MPI_Datatype datatype, int root,
                          struct ompi_communicator_t *comm, ompi_request_t ** request,
                          struct mca_coll_base_module_2_4_0_t *module, bool persistent)
{
  int rank, p, res, segsize;
  size_t size;
  BULLNBC_Schedule *schedule;
  enum { NBC_BCAST_LINEAR, NBC_BCAST_BINOMIAL, NBC_BCAST_CHAIN, NBC_BCAST_KNOMIAL} alg;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  if (1 == p) {
    return bullnbc_get_noop_request(persistent, request);
  }

  res = ompi_datatype_type_size(datatype, &size);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_size() (%i)", res);
    return res;
  }
  if(mca_coll_bullnbc_component.use_dynamic_rules) {
    if(0 != mca_coll_bullnbc_component.forced_params[BCAST].algorithm) {
      alg = mca_coll_bullnbc_component.forced_params[BCAST].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
      segsize = mca_coll_bullnbc_component.forced_params[BCAST].segsize;
      goto selected_rule;
    }
    if(bullnbc_module->com_rules[BCAST]) {
      int algorithm,dummy1,dummy2;
      algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[BCAST],
                                                           size * count, &dummy1, &segsize, &dummy2);
      if(algorithm) {
        alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
        goto selected_rule;
      }
    }
  }

  segsize = 16384;
  /* algorithm selection */
  if( bullnbc_ibcast_skip_dt_decision ) {
    if (p <= 4) {
      alg = NBC_BCAST_LINEAR;
    }
    else {
      alg = NBC_BCAST_BINOMIAL;
    }
  } else {
    if (p <= 4) {
      alg = NBC_BCAST_LINEAR;
    } else if (size * count < 65536) {
      alg = NBC_BCAST_BINOMIAL;
    } else if (size * count < 524288) {
      alg = NBC_BCAST_CHAIN;
      segsize = 8192;
    } else {
      alg = NBC_BCAST_CHAIN;
      segsize = 32768;
    }
  }

selected_rule:
  if(0 == segsize) {
    segsize = count * size; /* only one frag */
  }
  opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                      "Bullnbc ibcast : algorithm %d segmentsize %d",
                      alg + 1, segsize);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    switch(alg) {
      case NBC_BCAST_LINEAR:
        res = bcast_sched_linear(rank, p, root, schedule, buffer, count, datatype);
        break;
      case NBC_BCAST_BINOMIAL:
        res = bcast_sched_binomial(rank, p, root, schedule, buffer, count, datatype);
        break;
      case NBC_BCAST_CHAIN:
        res = bcast_sched_chain(rank, p, root, schedule, buffer, count, datatype, segsize, size);
        break;
      case NBC_BCAST_KNOMIAL:
        res = bcast_sched_knomial(rank, p, root, schedule, buffer, count, datatype, bullnbc_ibcast_knomial_radix);
        break;
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

int ompi_coll_bullnbc_ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
                            struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module)
{
    int res = nbc_bcast_init(buffer, count, datatype, root,
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

/* better binomial bcast
 * working principle:
 * - each node gets a virtual rank vrank
 * - the 'root' node get vrank 0
 * - node 0 gets the vrank of the 'root'
 * - all other ranks stay identical (they do not matter)
 *
 * Algorithm:
 * - each node with vrank > 2^r and vrank < 2^r+1 receives from node
 *   vrank - 2^r (vrank=1 receives from 0, vrank 0 receives never)
 * - each node sends each round r to node vrank + 2^r
 * - a node stops to send if 2^r > commsize
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
static inline int bcast_sched_binomial(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int maxr, vrank, peer, res;

  maxr = ceil_of_log2(p);

  RANK2VRANK(rank, vrank, root);

  /* receive from the right hosts  */
  if (vrank != 0) {
    for (int r = 0 ; r < maxr ; ++r) {
      if ((vrank >= (1 << r)) && (vrank < (1 << (r + 1)))) {
        VRANK2RANK(peer, vrank - (1 << r), root);
        res = NBC_Sched_recv (buffer, false, count, datatype, peer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }

    res = NBC_Sched_barrier (schedule);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  /* now send to the right hosts */
  for (int r = 0 ; r < maxr ; ++r) {
    if (((vrank + (1 << r) < p) && (vrank < (1 << r))) || (vrank == 0)) {
      VRANK2RANK(peer, vrank + (1 << r), root);
      res = NBC_Sched_send (buffer, false, count, datatype, peer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }
  }

  return OMPI_SUCCESS;
}

/* simple linear MPI_Ibcast */
static inline int bcast_sched_linear(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype) {
  int res;

  /* send to all others */
  if(rank == root) {
    for (int peer = 0 ; peer < p ; ++peer) {
      if (peer != root) {
        /* send msg to peer */
        res = NBC_Sched_send (buffer, false, count, datatype, peer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }
  } else {
    /* recv msg from root */
    res = NBC_Sched_recv (buffer, false, count, datatype, root, schedule, false);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
    }
  }

  return OMPI_SUCCESS;
}

/* simple chained MPI_Ibcast */
static inline int bcast_sched_chain(int rank, int p, int root, BULLNBC_Schedule *schedule, void *buffer, int count, MPI_Datatype datatype, int fragsize, size_t size) {
  int res, vrank, rpeer, speer, numfrag, fragcount;
  MPI_Aint ext;

  RANK2VRANK(rank, vrank, root);
  VRANK2RANK(rpeer, vrank-1, root);
  VRANK2RANK(speer, vrank+1, root);
  res = ompi_datatype_type_extent(datatype, &ext);
  if (MPI_SUCCESS != res) {
    NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
    return res;
  }

  if (count == 0) {
    return OMPI_SUCCESS;
  }

  numfrag = count * size/fragsize;
  if ((count * size) % fragsize != 0) {
    numfrag++;
  }

  fragcount = count/numfrag;

  for (int fragnum = 0 ; fragnum < numfrag ; ++fragnum) {
    char *buf = (char *) buffer + fragnum * fragcount * ext;
    int thiscount = fragcount;
    if (fragnum == numfrag-1) {
      /* last fragment may not be full */
      thiscount = count - fragcount * fragnum;
    }

    /* root does not receive */
    if (vrank != 0) {
      res = NBC_Sched_recv (buf, false, thiscount, datatype, rpeer, schedule, true);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    }

    /* last rank does not send */
    if (vrank != p-1) {
      res = NBC_Sched_send (buf, false, thiscount, datatype, speer, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }

      /* this barrier here seems awaward but isn't!!!! */
      if (vrank == 0)  {
        res = NBC_Sched_barrier (schedule);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          return res;
        }
      }
    }
  }

  return OMPI_SUCCESS;
}

/*
 * bcast_sched_knomial:
 *
 * Description: an implementation of Ibcast using k-nomial tree algorithm
 *
 * Time: (radix - 1)O(log_{radix}(comm_size))
 * Schedule length (rounds): O(log(comm_size))
 */
static inline int bcast_sched_knomial(
    int rank, int comm_size, int root, BULLNBC_Schedule *schedule, void *buf,
    int count, MPI_Datatype datatype, int knomial_radix)
{
    int res = OMPI_SUCCESS;

    /* Receive from parent */
    int vrank = (rank - root + comm_size) % comm_size;
    int mask = 0x1;
    while (mask < comm_size) {
        if (vrank % (knomial_radix * mask)) {
            int parent = vrank / (knomial_radix * mask) * (knomial_radix * mask);
            parent = (parent + root) % comm_size;
            res = NBC_Sched_recv(buf, false, count, datatype, parent, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            break;
        }
        mask *= knomial_radix;
    }
    mask /= knomial_radix;

    /* Send data to all children */
    while (mask > 0) {
        for (int r = 1; r < knomial_radix; r++) {
            int child = vrank + mask * r;
            if (child < comm_size) {
                child = (child + root) % comm_size;
                res = NBC_Sched_send(buf, false, count, datatype, child, schedule, false);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            }
        }
        mask /= knomial_radix;
    }

cleanup_and_return:
    return res;
}


static int nbc_bcast_inter_init(void *buffer, int count, MPI_Datatype datatype, int root,
                                struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int res;
  BULLNBC_Schedule *schedule;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  schedule = OBJ_NEW(BULLNBC_Schedule);
  if (OPAL_UNLIKELY(NULL == schedule)) {
    return OMPI_ERR_OUT_OF_RESOURCE;
  }

  if (root != MPI_PROC_NULL) {
    /* send to all others */
    if (root == MPI_ROOT) {
      int remsize;

      remsize = ompi_comm_remote_size (comm);

      for (int peer = 0 ; peer < remsize ; ++peer) {
        /* send msg to peer */
        res = NBC_Sched_send (buffer, false, count, datatype, peer, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          return res;
        }
      }
    } else {
      /* recv msg from root */
      res = NBC_Sched_recv (buffer, false, count, datatype, root, schedule, false);
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

int ompi_coll_bullnbc_ibcast_inter(void *buffer, int count, MPI_Datatype datatype, int root,
                                  struct ompi_communicator_t *comm, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_bcast_inter_init(buffer, count, datatype, root,
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

int ompi_coll_bullnbc_bcast_init(void *buffer, int count, MPI_Datatype datatype, int root,
                                struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_bcast_init(buffer, count, datatype, root,
                             comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int
ompi_coll_bullnbc_pbcast_init(void *buff,
                                  size_t parts, size_t count,
                                  struct ompi_datatype_t *datatype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module)
{
    int ret = OMPI_SUCCESS, comm_size, rank;
    ompi_coll_bullnbc_pcoll_request_t * req;

    rank = ompi_comm_rank (comm);
    comm_size = ompi_comm_size (comm);

    if (1 == comm_size) {
        NBC_DEBUG(10,"Bcast alone, nothing to do\n");
        return bullnbc_get_noop_request(true, request);
    }

    ompi_coll_bullnbc_module_t *mod = (ompi_coll_bullnbc_module_t*) module;
    module_need_progress(mod);

    req = ompi_mca_coll_bullnbc_alloc_pcoll_request();

    req->comm = comm;
    req->tag = ompi_coll_base_nbc_reserve_tags(comm, 1);

    req->total_sparts = 0;
    req->total_rparts = 0;
    req->user_sreqs = MPIX_NO_REQUESTS;
    req->user_rreqs = MPIX_NO_REQUESTS;

    req->n_internal_reqs = (rank == root) ? comm_size - 1 : 1;
    req->internal_reqs = bullnbc_xmalloc(req->n_internal_reqs * sizeof(ompi_request_t *));

    if (rank == root) {
        for(int i = 1 ; i < comm_size ; ++i){
            ret = mca_part.part_psend_init(buff, parts, count, datatype,
                                           (rank+i)%comm_size,
                                           req->tag, comm, info,
                                           &req->internal_reqs[i - 1]);
        }

        req->send_subparts = bullnbc_xmalloc(parts * sizeof(opal_list_t));
        req->recv_subparts = bullnbc_xmalloc(parts * sizeof(opal_list_t));

        for (size_t part=0; part<parts; ++part){
            OBJ_CONSTRUCT(&req->send_subparts[part], opal_list_t);
            OBJ_CONSTRUCT(&req->recv_subparts[part], opal_list_t);
            for(int i = 1 ; i < comm_size ; ++i){
                ompi_coll_bullnbc_subpart* subpart;
                subpart = (ompi_coll_bullnbc_subpart*) opal_free_list_wait(&subpart_free_list);
                subpart->sub_req = req->internal_reqs[i-1];
                subpart->part_idx = part;
                opal_list_append(&req->send_subparts[part],&subpart->super.super);
            }

            NBC_DEBUG(1,"Bcast %p Recv subpart list %d (%p) contains %d elems\n",
            req, part, &req->send_subparts[part],
            opal_list_get_size(&req->send_subparts[part]));
        }
    } else {
        ret = mca_part.part_precv_init(buff, parts, count, datatype, root,
                                       req->tag, comm, info,
                                       &req->internal_reqs[0]);

        req->send_subparts = NULL;
        req->recv_subparts = bullnbc_xmalloc(parts * sizeof(opal_list_t));
        for (size_t part=0; part<parts; ++part){
            OBJ_CONSTRUCT(&req->recv_subparts[part], opal_list_t);
            ompi_coll_bullnbc_subpart* subpart;
            subpart = (ompi_coll_bullnbc_subpart*) opal_free_list_wait(&subpart_free_list);
            subpart->sub_req = req->internal_reqs[0];
            subpart->part_idx = part;
            opal_list_append(&req->recv_subparts[part],&subpart->super.super);
        }
    }

    if (OMPI_SUCCESS != ret) {
        goto fail_to_init_sub_req;
    }

    *request = &req->req_ompi;

    NBC_DEBUG(10, "Bcast init succeed\n");
    return OMPI_SUCCESS;

fail_to_init_sub_req:
    NBC_Error("Bcast failed to init\n");
    ompi_mca_coll_bullnbc_free((ompi_request_t **)&req);
    return ret;
}


int ompi_coll_bullnbc_bcast_inter_init(void *buffer, int count, MPI_Datatype datatype, int root,
                                      struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_bcast_inter_init(buffer, count, datatype, root,
                                   comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

/* Root schedule a send per peer */
static int
ompi_coll_bullnbc_pbcastr_linear(void *buff,
                                 size_t parts, size_t count,
                                 struct ompi_datatype_t *datatype,
                                 ompi_request_t ** subreqs,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 struct ompi_info_t *info,
                                 ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;
    int ncomm, nrecv;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
        ncomm = comm_size -1;
        nrecv = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
        ncomm = 1;
        nrecv = 1;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   ncomm, nrecv,
                                                   comm, request);

    if(root == rank) {
        pcoll_dag_item** nodes = bullnbc_xmalloc(2*parts * sizeof(pcoll_dag_item*));
        for (int part=0; part < parts; ++part){
            nodes[part] = sched_ready(part, req, comm_size - 1);
            nodes[parts + part] = sched_complete(part, req);
        }
        for (int vpeer = 1; vpeer < comm_size; ++vpeer) {
            pcoll_dag_item* send;
            send = sched_send(buff, count*parts, datatype,
                              (rank + vpeer) % comm_size, comm,
                              &req->internal_reqs[vpeer-1], req,
                              parts);
            for (int part=0; part < parts; ++part){
                    create_dependency(nodes[part], send);
                    create_dependency(send, nodes[parts + part]);
            }
        }
        free(nodes);
    } else {
        pcoll_dag_item* recv;
        recv = sched_recv(buff, count*parts, datatype,
                          root, comm,
                          &req->internal_reqs[0], req,
                          0,
                          parts);
        for (int part=0; part < parts; ++part){
            pcoll_dag_item* complete = sched_complete(part, req);
            create_dependency(recv, complete);
        }
    }
    return OMPI_SUCCESS;
}

/* Use this tree mapping on ranks (written in base 2):
 * 0--\--\---------\-------\
 * |   \  \         \       \
 * 1   10 100-\    1000      \
 *     |   |   \     | \      \
 *     11 101  110 1001 1010  1100-\
 *              |         |     |   \
 *             111      1011  1101 1110
 */
static int
ompi_coll_bullnbc_pbcastr_binomial(void *buff,
                                   size_t parts, size_t count,
                                   struct ompi_datatype_t *datatype,
                                   ompi_request_t ** subreqs,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank, vrank;
    int nchild = 0;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    vrank = (rank - root + comm_size) % comm_size;

    while ((vrank + (1<<nchild) < comm_size) &&
           (!(vrank & (1<<nchild)))) {
        nchild++;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;
    int ncomm, nrecv;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
        ncomm = nchild;
        nrecv = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
        ncomm = nchild +1;
        nrecv = 1;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   ncomm, nrecv,
                                                   comm, request);

    pcoll_dag_item** nodes = bullnbc_xmalloc(2*parts * sizeof(pcoll_dag_item*));
    if (rank == root) {
        for (int part=0; part < parts; ++part){
            nodes[part] = sched_ready(part, req, nchild);
        }
    }
    for (int part=0; part < parts; ++part){
        nodes[parts + part] = sched_complete(part, req);
    }

    pcoll_dag_item** sends = bullnbc_xmalloc(nchild * sizeof(pcoll_dag_item*));
    for (int child=nchild-1; child >=0 ; --child) {
        pcoll_dag_item* send;
        send = sched_send(buff, count*parts, datatype,
                          (root + vrank + (1<<child)) % comm_size,
                          comm,
                          &req->internal_reqs[child], req,
                          parts);
        sends[child] = send;
        if(0 == vrank){
            /* Link ready to send on root */
            for (int part=0; part < parts; ++part){
                create_dependency(nodes[part], send);
            }
        }
        /* Link send to complete on non root */
        for (int part=0; part < parts; ++part){
            create_dependency(send, nodes[parts+part]);
        }
    }
    if (vrank) {
        int prank = 1<<nchild;
        while (!(vrank & prank)) {
            prank <<=1;
        }
        prank = vrank - prank;

        pcoll_dag_item* recv;
        recv = sched_recv(buff, count*parts, datatype,
                          (prank+root)%comm_size , comm,
                          &req->internal_reqs[nchild], req,
                          0,
                          parts + nchild);
        /* Link recv to complete on non root */
        for (int part=0; part < parts; ++part){
            create_dependency(recv, nodes[parts+part]);
        }
        for (int child=nchild-1; child >=0 ; --child) {
            create_dependency(recv, sends[child]);
        }
    }
    free(nodes);
    free(sends);
    return OMPI_SUCCESS;
}

/* Same algorithm but all partitions progresses independantly */
static int
ompi_coll_bullnbc_pbcastr_binomial_partitioned(void *buff,
                                               size_t parts, size_t count,
                                               struct ompi_datatype_t *datatype,
                                               ompi_request_t ** subreqs,
                                               int root,
                                               struct ompi_communicator_t *comm,
                                               struct ompi_info_t *info,
                                               ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank, vrank;
    int nchild = 0;

    ptrdiff_t lb, extent;
    size_t datatype_size;
    ompi_datatype_get_extent(datatype, &lb, &extent);
    ompi_datatype_type_size(datatype, &datatype_size);

    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    vrank = (rank - root + comm_size) % comm_size;

    while ((vrank + (1<<nchild) < comm_size) &&
           (!(vrank & (1<<nchild)))) {
        nchild++;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;
    int ncomm, nrecv;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
        ncomm = parts * nchild;
        nrecv = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
        ncomm = parts * (nchild +1);
        nrecv = parts;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   ncomm, nrecv,
                                                   comm, request);

    pcoll_dag_item** nodes = bullnbc_xmalloc(2*parts * sizeof(pcoll_dag_item*));
    if (rank == root) {
        for (int part=0; part < parts; ++part){
            nodes[part] = sched_ready(part, req, nchild);
        }
    }
    for (int part=0; part < parts; ++part){
        nodes[parts + part] = sched_complete(part, req);
    }

    pcoll_dag_item** sends = bullnbc_xmalloc(nchild * sizeof(pcoll_dag_item*));
    for (int part = 0; part < parts; ++part) {
        for (int child=nchild-1; child >=0 ; --child) {
            pcoll_dag_item* send;
            send = sched_send(((char*)buff) + part*count*extent, count, datatype,
                              (root + vrank + (1<<child)) % comm_size,
                              comm,
                              &req->internal_reqs[parts *child + part], req,
                              1);
            sends[child] = send;
            if(0 == vrank){
                /* Link ready to send on root */
                create_dependency(nodes[part], send);
            }
            /* Link send to complete on non root */
            create_dependency(send, nodes[parts+part]);
        }
        if (vrank) {
            int prank = 1<<nchild;
            while (!(vrank & prank)) {
                prank <<=1;
            }
            prank = vrank - prank;

            pcoll_dag_item* recv;
            recv = sched_recv(((char*)buff) + part*count* extent, count, datatype,
                              (prank+root)%comm_size , comm,
                              &req->internal_reqs[parts*nchild + part], req,
                              part,
                              1 + nchild);
            /* Link recv to complete on non root */
            create_dependency(recv, nodes[parts+part]);
            for (int child=nchild-1; child >=0 ; --child) {
                create_dependency(recv, sends[child]);
            }
        }
    }
    free(nodes);
    free(sends);
    return OMPI_SUCCESS;
}
static int
ompi_coll_bullnbc_pbcastr_chain(void *buff,
                                size_t parts, size_t count,
                                struct ompi_datatype_t *datatype,
                                ompi_request_t ** subreqs,
                                int root,
                                struct ompi_communicator_t *comm,
                                struct ompi_info_t *info,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    int vrank = (rank - root + comm_size) % comm_size;
    int nsend = (vrank +1 == comm_size ? 0: 1);

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;
    int ncomm, nrecv;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
        ncomm = nsend;
        nrecv = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
        ncomm = nsend +1;
        nrecv = 1;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   ncomm, nrecv,
                                                   comm, request);

    pcoll_dag_item** completes = bullnbc_xmalloc(parts * sizeof(pcoll_dag_item*));
    for (int part=0; part < parts; ++part){
        completes[part] = sched_complete(part, req);
    }
    pcoll_dag_item* send = NULL;
    int n_comm = 0;
    if(vrank+1 < comm_size) {
        send = sched_send(buff, count*parts, datatype,
                          (rank + 1) % comm_size, comm,
                          &req->internal_reqs[n_comm++], req,
                          parts);
        if (root == rank) {
            for (int part=0; part < parts; ++part){
                pcoll_dag_item* ready = sched_ready(part, req, 1);
                create_dependency(ready, send);
            }
        }

        for (int part=0; part < parts; ++part){
            create_dependency(send, completes[part]);
        }
    }
    if (vrank) {
        pcoll_dag_item* recv;
        recv = sched_recv(buff, count*parts, datatype,
                          (rank -1 + comm_size)% comm_size, comm,
                          &req->internal_reqs[n_comm++], req,
                          0,
                          parts+ nsend);
        if (send) {
            create_dependency(recv, send);
        }
        for (int part=0; part < parts; ++part){
            create_dependency(recv, completes[part]);
        }
    }
    free(completes);
    return OMPI_SUCCESS;
}


/* Use this tree mapping on ranks (written in base K):
 * 0-------------\--------------\------\--\-------\
 * |              \              \      \  \       \
 * 1-----\         2------\  ..  k-1    10  20 ... k-1|0
 * |  \   \        | \     \
 * 11 21 .. k-1|1  12 22 .. k-1|2
 * In base K the rank can be wirtten as X[0]*YYYY with 0<X<K, YYYY the parent
 * and the number of 0 corresponds to the step at which the parent sent the data.
 */
static int
ompi_coll_bullnbc_pbcastr_knomial(void *buff,
                                   size_t parts, size_t count,
                                   struct ompi_datatype_t *datatype,
                                   ompi_request_t ** subreqs,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank, vrank;
    int nchild = 0;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    vrank = (rank - root + comm_size) % comm_size;
    int radix = bullnbc_ibcast_knomial_radix;

    int nextpow = 1;
    /* nextpow is the next power of radix after vrank */
    while (vrank / nextpow) {
        nextpow *= radix;
    }

    /* Look for a upper bound of the number of children*/
    int max_children = radix, powafter = 1;
    while(powafter * nextpow < comm_size) {
        powafter *= radix;
        max_children += radix;
    }
    int * vchildren = bullnbc_xmalloc(max_children * sizeof(int));

    int mask = 1;
    int i = 1;
    /* Test each potential child exists */
    while (i*mask*nextpow  + vrank < comm_size) {
        vchildren[nchild]= i*mask*nextpow  + vrank;
        nchild ++;
        i++;
        if (i == radix) {
            i = 1;
            mask *= radix;
        }
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;
    int ncomm, nrecv;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
        ncomm = nchild;
        nrecv = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
        ncomm = nchild +1;
        nrecv = 1;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   ncomm, nrecv,
                                                   comm, request);

    pcoll_dag_item** nodes = bullnbc_xmalloc(2*parts * sizeof(pcoll_dag_item*));
    if (rank == root) {
        for (int part=0; part < parts; ++part){
            nodes[part] = sched_ready(part, req, nchild);
        }
    }
    for (int part=0; part < parts; ++part){
        nodes[parts + part] = sched_complete(part, req);
    }

    pcoll_dag_item** sends = bullnbc_xmalloc(nchild * sizeof(pcoll_dag_item*));
    mask = i = 1;
    /* Send to each child */
    for (int child = 0; child < nchild ; ++child) {
        int vpeer = vchildren[child];
        NBC_DEBUG(10, "vrank %d send to %d = %d x %d|%d\n",
                vrank, vpeer, i, mask, vrank);
        pcoll_dag_item* send;
        send = sched_send(buff, count*parts, datatype,
                          (vpeer + root) % comm_size,
                          comm,
                          &req->internal_reqs[child], req,
                          parts);
        sends[child] = send;
        if(0 == vrank){
            /* Link ready to send on root */
            for (int part=0; part < parts; ++part){
                create_dependency(nodes[part], send);
            }
        }
        /* Link send to complete on non root */
        for (int part=0; part < parts; ++part){
            create_dependency(send, nodes[parts+part]);
        }

        i++;
        if (i == radix) {
            i = 1;
            mask *= radix;
        }
    }

    if (vrank) {
        /* Schedule a recv from the parent which has the same rank
         * as the current process except the highest digit of the local rank, */
        int high_digit = nextpow / radix;
        high_digit = vrank / high_digit * high_digit;
        int prank = vrank - high_digit;

        pcoll_dag_item* recv;
        recv = sched_recv(buff, count*parts, datatype,
                          (prank+root)%comm_size , comm,
                          &req->internal_reqs[nchild], req,
                          0,
                          parts + nchild);
        /* Link recv to complete on non root */
        for (int part=0; part < parts; ++part){
            create_dependency(recv, nodes[parts+part]);
        }
        for (int child = 0; child < nchild ; ++child) {
            create_dependency(recv, sends[child]);
        }
    }

    free(vchildren);
    free(nodes);
    free(sends);
    return OMPI_SUCCESS;
}


/*
 * This routine creates a DAG for a bcast scatter-allgather.
 *
 * This is inspired from coll/basic component implementation.
 * This is equivalent to a binomial MPI_scatter of the data followed by
 * a hypercube MPI_Allgather of all the fragments. At iteration k, each rank
 * send 2^k * size/comm_size data to (rank XOR 2^k), called remote.
 * In case of comm_size is not a power of 2, some ranks do not have remote.
 * Binomial bcasts of maximal size of 2^k are then done.
 */
static int
ompi_coll_bullnbc_pbcastr_scatter_allgather(void *buff,
                                            size_t parts, size_t count,
                                            struct ompi_datatype_t *datatype,
                                            ompi_request_t ** subreqs,
                                            int root,
                                            struct ompi_communicator_t *comm,
                                            struct ompi_info_t *info,
                                            ompi_request_t ** request,
                                            struct mca_coll_base_module_2_4_0_t *module)
{

    int comm_size, rank, vrank;

    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    vrank = (rank - root + comm_size) % comm_size;

    size_t datatype_size;
    ptrdiff_t lb, extent;
    ompi_datatype_get_extent(datatype, &lb, &extent);
    ompi_datatype_type_size(datatype, &datatype_size);

    /* CORNER CASES */

    /* TODO Heterogenous datatypes sizes */

    /* PRECOMPUTE REQUIRED COMMS (to alloc request) */

    int comm_mask = 0x1;
    int scatter_parent = -1;
    int n_recv_scatter = vrank ? 1: 0;
    int n_send_scatter = 0;

    /* Compute scatter children and parent */
    while (vrank + comm_mask < comm_size){
        if (vrank & comm_mask) {
            scatter_parent = (vrank - comm_mask + root) % comm_size;
            break;
        }
        n_send_scatter ++;
        comm_mask <<=1;
    }

    if (vrank && -1 == scatter_parent ) {
        /* Scatter parent may be unknown if my rank is near comm_size */
        while (0 == (vrank & comm_mask)) {
            comm_mask <<=1;
        }
        scatter_parent = (vrank - comm_mask + root) % comm_size;
    }

    NBC_DEBUG(5, "%d send in scatter and parent is %d\n",
              n_send_scatter, scatter_parent);
    /* Size of the sub tree in which vrank is root */
    int local_tree_size = 1<<n_send_scatter;
    if ( vrank + local_tree_size > comm_size ) {
        local_tree_size = comm_size - vrank;
    }

    /* Precompute the count of send recv for scatter */
    int n_send_allgather = 0;
    int n_extra_send_allgather = 0;
    int n_recv_allgather = 0;
    int n_extra_recv_allgather = 0;
    for (comm_mask = 0x1; comm_mask < comm_size; comm_mask <<=1) {
        int remote = vrank ^ comm_mask;
        if (remote < comm_size) {
            n_send_allgather ++;
            n_recv_allgather ++;
        }

        /* Case comm_size is not a power of two:
         * supply data to peers in local subtree that do have remote.
         * Peer are iterated in a binomial tree pattern in case they
         * also have to propagate missing data. */
        int local_root = vrank / comm_mask * comm_mask;
        int remote_root = remote / comm_mask * comm_mask;
        if (remote_root + comm_mask > comm_size) {
            /* remote tree is incomplete, so mine is complete */
            int nprocs_remote = comm_size - remote_root;
            /* Some peer on local tree may not have a remote and need data */
            for (int sub_mask = comm_mask >>1; sub_mask ; sub_mask >>=1) {
                int vpeer =  vrank ^ sub_mask;
                int backup_tree_root = vrank / (sub_mask <<1) * (sub_mask << 1);
                if ((vrank < vpeer) && 
                    (vrank < backup_tree_root + nprocs_remote) &&
                    (vpeer >= backup_tree_root + nprocs_remote)) {
                    n_extra_send_allgather ++; /* Send missing data */
                } else if ((vpeer < vrank) && 
                    (vpeer < backup_tree_root + nprocs_remote) &&
                    (vrank >= backup_tree_root + nprocs_remote)) {
                    n_extra_recv_allgather ++; /* Recv missing data */
                }
            }
        }
    }

    int n_send = n_send_scatter + n_send_allgather + n_extra_send_allgather;
    int n_recv = n_recv_scatter + n_recv_allgather + n_extra_recv_allgather;
    NBC_DEBUG(10, "%d(=%d+%d+%d) send %d(=%d+%d+%d) recv\n",
              n_send, n_send_scatter, n_send_allgather, n_extra_send_allgather,
              n_recv, n_recv_scatter, n_recv_allgather, n_extra_recv_allgather);

    /* ALLOC REQUEST */

    ompi_coll_bullnbc_pcoll_request_t * req;
    ompi_request_t ** sendreqs, **recvreqs;
    int n_sendreqs, n_recvreqs;

    if (rank == root) {
        sendreqs = subreqs;
        n_sendreqs = parts;
        recvreqs = MPIX_NO_REQUESTS;
        n_recvreqs = 0;
    } else {
        sendreqs = MPIX_NO_REQUESTS;
        n_sendreqs = 0;
        recvreqs = subreqs;
        n_recvreqs = parts;
    }
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(n_sendreqs, sendreqs,
                                                   n_recvreqs, recvreqs,
                                                   n_send + n_recv, n_recv,
                                                   comm, request);

    pcoll_dag_item** ready, **completions;
    ready = bullnbc_xmalloc((2*parts) * sizeof(pcoll_dag_item*));
    completions = ready + parts;

    if (rank == root) {
        for (int part=0; part < parts; ++part){
            ready[part] = sched_ready(part, req, n_send_scatter + 1);
        }
    }
    for (int part=0; part < parts; ++part){
        completions[part] = sched_complete(part, req);
    }

    /************************************
     * ADD SCATTER COMMS TO THE REQUEST */

    /* Array describing how data are scattered across ranks */
    int * block_offset = bullnbc_xmalloc ((comm_size+1) * sizeof(int));
    int rest_count = count % comm_size;
    for (int i=0 ; i <= comm_size ;++i) {
        block_offset[i] =  count / comm_size * i;
        block_offset[i] += i < rest_count ? i : rest_count;
        block_offset[i] *= parts;
        /* TODO split comms per part and run them in parallel */
    }

    /* Send order must be preserved for a given peer to avoid crossing
     * comms. This tracks the last send to each peer to add dependencies */
    int n_send_peer = comm->c_cube_dim + 1;
    pcoll_dag_item ** last_sends = bullnbc_xmalloc(n_send_peer * sizeof(pcoll_dag_item*));
    for (int i = 0; i < n_send_peer; ++i) {
        last_sends[i] = NULL;
    }
    /* All data from any previous receive will be used for next iterations.
     * Store recv for dependencies */
    pcoll_dag_item ** prev_recvs = bullnbc_xmalloc(n_recv * sizeof(pcoll_dag_item*));
    int n_prev_recv = 0;


    int n_ready = vrank ? 0: parts;
    int recv_req_idx = 0;
    int send_req_idx = 0;
    if (vrank) {
        int recv_count = block_offset[vrank + local_tree_size];
        recv_count -= block_offset[vrank];

        /* Scatter: recv a part of the data */
        pcoll_dag_item* recv;
        NBC_DEBUG(10, "[scatter] recv %d+%d from %d (req %d)\n",
                  vrank, recv_count, scatter_parent, n_send);
        recv = sched_recv(((char*)buff) + block_offset[vrank]*extent,
                          recv_count, datatype,
                          scatter_parent, comm,
                          &req->internal_reqs[n_send + recv_req_idx], req,
                          n_ready + recv_req_idx,
                          parts + n_send);
        recv_req_idx++; /* TODO Store this in requests */
        prev_recvs[n_prev_recv++] = recv;
    }

    for (int mask_idx = n_send_scatter -1; mask_idx >= 0; --mask_idx) {
        int mask = 1<< mask_idx;
        int vchild = vrank + mask;
        int send_next_rank =  vchild + mask;
        if (send_next_rank > comm_size) {
            send_next_rank = comm_size;
        }
        int send_count = block_offset[send_next_rank] - block_offset[vchild];

        pcoll_dag_item* send;
        NBC_DEBUG(10, "[scatter] send 0x%x\n", mask);
        send = sched_send(((char*)buff) + block_offset[vchild]*extent,
                          send_count, datatype,
                          (root + vchild) % comm_size,
                          comm,
                          &req->internal_reqs[send_req_idx++],
                          req,
                          parts +1); /*completion + a send ? */
        if (vrank) {
            /* I am not root, so I need to recv data before any send */
            create_dependency(prev_recvs[0], send);
        } else {
            /* Root need readys to trigger all scatter phase send */
            for (int part=0; part < parts; ++part){
                create_dependency(ready[part], send);
            }
        }
        last_sends[mask_idx] = send;
    }

    /**************************************
     * ADD ALLGATHER COMMS TO THE REQUEST */

    /* Note the usage of 'peer' and 'remote':
     * remote is the process 'rank XOR 2^k' that should provide data to me
     * a peer is a process of rank XOR 2^j, j<k that should received this same
     * data at this iteration.
     * Peers are used for the bcast backup if a remote is missing */

    for (int mask_idx = 0x1, mask = 0x1; mask < comm_size; mask <<=1, ++mask_idx) {
        int vremote = vrank ^ mask;
        int remote = (vremote + root) % comm_size;
        int base_rank = vrank / mask * mask;
        int base_remote = vremote / mask * mask;
        int someone_needs_remote = (base_remote + mask > comm_size) ;

        int data_end_remote = base_remote + mask;
        if (data_end_remote > comm_size){
            data_end_remote = comm_size;
        }
        int data_end_rank = base_rank + mask;
        if (data_end_rank > comm_size) {
            data_end_rank = comm_size;
        }
        int send_count = block_offset[data_end_rank] - block_offset[base_rank];
        int recv_count = block_offset[data_end_remote] - block_offset[base_remote];

        /* Sendrecv to my remote */
        if (vremote < comm_size) {
            pcoll_dag_item* send, *recv;
            NBC_DEBUG(10, "[gather] recv 0x%x of %d (req%d)\n",
                      mask, recv_count, recv_req_idx);
            recv = sched_recv(((char*)buff) + block_offset[base_remote]*extent,
                              recv_count, datatype,
                              remote, comm,
                              &req->internal_reqs[n_send + recv_req_idx], req,
                              n_ready + recv_req_idx,
                              parts + n_send_allgather + n_extra_send_allgather);
            recv_req_idx++;

            NBC_DEBUG(10, "[gather] send 0x%x\n", mask);
            send = sched_send(((char*)buff) + block_offset[base_rank]*extent,
                              send_count, datatype,
                              remote,
                              comm,
                              &req->internal_reqs[send_req_idx++], req,
                              parts +1); /*completion + a send ? */
            for (int part=0; part < parts; ++part){
                create_dependency(recv, completions[part]);
                create_dependency(send, completions[part]);
            }
            for (int i = 0; i < n_prev_recv; ++i) {
                create_dependency(prev_recvs[i], send);
            }

            if (last_sends[mask_idx]) {
                create_dependency(last_sends[mask_idx], send);
            }
            last_sends[mask_idx] = send;

            if (0 == vrank && 0x1 == mask) {
                for (int part=0; part < parts; ++part){
                    create_dependency(ready[part], send);
                }
            }

            prev_recvs[n_prev_recv++] = recv;
        }

        if (someone_needs_remote) {
            /* Add smaller binomial bcasts to provide to ranks in range of
             * mask missing blocks. It starts on a rank that have remote.
             * This approach prevents from creating imbalance at the end
             * by including extra comms in between usual ones */
            int nprocs_remote = comm_size - base_remote;
            for (int bcast_mask = mask >> 1, idx=mask_idx-1; bcast_mask > 0; bcast_mask >>=1, --idx) {
                /* If some remotes are missing, all my local peers exists */
                int bcast_root = vrank / (bcast_mask*2) * (bcast_mask*2);
                /* At each loop, bcast_root is garanteed to have the missing data.
                 * and if needed it will send it one rank closer to any procs.
                 * This propagation can be defined like a moving frontier.
                 * Just wait that frontier to be on a local link */
                int data_frontier_rank = bcast_root + nprocs_remote;

                /* Search for peers that does not have remote */
                int vpeer = vrank ^ bcast_mask;
                int peer = (vpeer + root) % comm_size;
                /* In my part of the tree, only smallest ranks may have the
                 * mising data */
                if ((vpeer < vrank)
                    && (vrank >= data_frontier_rank)
                    && (vpeer < data_frontier_rank)) {
                    /* Happen always at most once: when the N least significant
                     * bits of vrank are bigger (or equal) than nprocs_remote but N-1 bits
                     * of vranks are lower. Occurs at mask=1<<N.
                     * If it never occur, it means vrank < nprocs which implies
                     * that vrank has a remote and therefore it aleady has the data */
                    pcoll_dag_item* recv;
                    NBC_DEBUG(10, "[extra] recv from %d (data %d +%d)\n",
                              peer, base_remote,recv_count);
                    recv = sched_recv(((char*)buff) + block_offset[base_remote]*extent,
                                      recv_count, datatype,
                                      peer, comm,
                                      &req->internal_reqs[n_send + recv_req_idx], req,
                                      n_ready + recv_req_idx,
                                      parts + n_send_allgather + n_extra_send_allgather);
                    recv_req_idx++;
                    prev_recvs[n_prev_recv++] = recv;
                    for (int part=0; part < parts; ++part){
                        create_dependency(recv, completions[part]);
                    }
                }
                if ((vrank < vpeer)
                    && (vrank < data_frontier_rank)
                    && (vpeer >= data_frontier_rank)) {

                    pcoll_dag_item* send;
                    /* My peer needs remote data got from a previous peer
                     * or by my remote */
                    NBC_DEBUG(10, "[extra] send to %d (data %d +%d)\n",
                              peer, base_remote, recv_count);
                    send = sched_send(((char*)buff) + block_offset[base_remote]*extent,
                                      recv_count, datatype,
                                      peer,
                                      comm,
                                      &req->internal_reqs[send_req_idx++], req,
                                      parts +1); /*completion + a send ? */
                    for (int i = 0; i < n_prev_recv; ++i) {
                        create_dependency(prev_recvs[i], send);
                    }
                    for (int part=0; part < parts; ++part){
                        create_dependency(send, completions[part]);
                    }
                    if (last_sends[idx]) {
                        create_dependency(last_sends[idx], send);
                    }
                    last_sends[idx] = send;
                }
            }
        }
    }

    free(ready);
    free(block_offset);
    free(last_sends);
    free(prev_recvs);
    return OMPI_SUCCESS;
}

int
ompi_coll_bullnbc_pbcastr_init(void *buff,
                               size_t parts, size_t count,
                               struct ompi_datatype_t *datatype,
                               ompi_request_t ** subreqs,
                               int root,
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

    if (comm_size < 2 || datatype_size == 0 || 0 == count ) {
        ompi_mca_coll_bullnbc_pcoll_init_empty_req(parts, subreqs,
                                                       parts, subreqs,
                                                       comm, request);
        return MPI_SUCCESS;
    }

    int alg = pbcastr_algorithm;
    if (0 == alg) {
        if (mca_coll_bullnbc_uniform_partition_sizes) {
            alg = 6;
        } else {
            alg = 2;
        }
    }

    if (! mca_coll_bullnbc_uniform_partition_sizes && 6 == alg){
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Non uniform partition sizes", true,
                       6, "bcast", 2);
        alg = 2;
    }

    module_need_progress((ompi_coll_bullnbc_module_t*) module);

    NBC_DEBUG(1, "Use algorithm %d for pbcastr with %d parts, count %d and root %d\n",alg, parts, count, root);
    switch (alg){
        case 1:
            return ompi_coll_bullnbc_pbcastr_linear(buff, parts, count,
                                                    datatype, subreqs,
                                                    root, comm, info,
                                                    request, module);
        case 2:
            return ompi_coll_bullnbc_pbcastr_binomial(buff, parts, count,
                                                      datatype, subreqs,
                                                      root, comm, info,
                                                      request, module);
        case 3:
            return ompi_coll_bullnbc_pbcastr_chain(buff, parts, count,
                                                   datatype, subreqs,
                                                   root, comm, info,
                                                   request, module);
        case 4:
            return ompi_coll_bullnbc_pbcastr_knomial(buff, parts, count,
                                                     datatype, subreqs,
                                                     root, comm, info,
                                                     request, module);
        case 5:
            return ompi_coll_bullnbc_pbcastr_scatter_allgather(buff, parts, count,
                                                               datatype, subreqs,
                                                               root, comm, info,
                                                               request, module);
        case 6:
            return ompi_coll_bullnbc_pbcastr_binomial_partitioned(buff, parts, count,
                                                                  datatype, subreqs,
                                                                  root, comm, info,
                                                                  request, module);
        default:
            fprintf(stderr, "Pbcastr algorithm %d is not implemented yet\n", alg); fflush(stderr);
            abort();
    }
}
