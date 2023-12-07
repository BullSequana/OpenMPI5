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
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
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
#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_partitioned_dag.h"
#include "opal/util/show_help.h"

static inline int
gather_linear(int rank, int p, int root, BULLNBC_Schedule *schedule,
              const void * sendbuf, int sendcount, MPI_Datatype sendtype,
              void * recvbuf, int recvcount, MPI_Datatype recvtype,
              int inplace );

static inline int
gather_binomial(int rank, int p, int root, BULLNBC_Schedule *schedule,
                const void * sendbuf, int scount, MPI_Datatype sendtype,
                void * recvbuf, int rcount, MPI_Datatype recvtype,
                int inplace, void ** tmpbuf_ptr);

static mca_base_var_enum_value_t gather_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binomial"},
    {0, NULL}
};
static mca_base_var_enum_value_t pgatherr_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binomial"},
    {3, "linear_part"},
    {0, NULL}
};
static int pgatherr_algorithm;
int mca_coll_bullnbc_gather_uniform_partition_sizes = 0;

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_gather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != gather_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "igather_algorithm_count",
                                         "Number of gather algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_gather_algorithms", gather_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "igather_algorithm",
                                         "Which gather algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "igather_algorithm_segmentsize",
                                  "Segment size in bytes used by default for igather algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  pgatherr_algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_pgatherr_algorithms",
                                  pgatherr_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "pgatherr_algorithm",
                                         "Which pgatherr algorithm is used",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0,
                                         MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &pgatherr_algorithm);
  OBJ_RELEASE(new_enum);
  /* TODO allgather to detect the  case of some coll does and some doesn't fit with this hint */
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "gather_uniform_partition_count",
                                  "Hint that all peers of a given pgather communication use the same count of partitions for send and recv. Enabled finer send/recv optimizations",
                                  MCA_BASE_VAR_TYPE_BOOL, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_coll_bullnbc_gather_uniform_partition_sizes);

  return OMPI_SUCCESS;
}

static int nbc_gather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                           int recvcount, MPI_Datatype recvtype, int root,
                           struct ompi_communicator_t *comm, ompi_request_t ** request,
                           struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  int rank, p, res;
  size_t size;
  BULLNBC_Schedule *schedule;
  char inplace = 0;
  void *tmpbuf = NULL;
  enum {IGATHER_LINEAR, IGATHER_BINOMIAL} alg;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

  rank = ompi_comm_rank (comm);
  p = ompi_comm_size (comm);

  if (root == rank) {
    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  }

  if (inplace) {
    sendcount = recvcount;
    sendtype = recvtype;
  }

  res = ompi_datatype_type_size (sendtype, &size);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
  }
  size *= sendcount;

  if(mca_coll_bullnbc_component.use_dynamic_rules) {
      if(0 != mca_coll_bullnbc_component.forced_params[GATHER].algorithm) {
          /* if op is not commutative or MPI_IN_PLACE was specified we have to deal with it */
          alg = mca_coll_bullnbc_component.forced_params[GATHER].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
          goto selected_rule;
      }
      if(bullnbc_module->com_rules[GATHER]) {
          int algorithm,dummy1,dummy2,dummy3;
          algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[GATHER],
                      size, &dummy1, &dummy2, &dummy3);
          if(algorithm) {
              alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
              goto selected_rule;
          }
      }
  }
  if (size < 1024){
      alg = IGATHER_BINOMIAL;
  } else {
      alg = IGATHER_LINEAR;
  }
selected_rule:
  opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                      "Bullnbc igather : algorithm %d (no segmentation supported)",
                      alg + 1);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      free(tmpbuf);
      return OMPI_ERR_OUT_OF_RESOURCE;
    }
    switch(alg){
        case IGATHER_LINEAR:
            res = gather_linear(rank, p, root, schedule,
                                sendbuf, sendcount, sendtype,
                                recvbuf, recvcount, recvtype, inplace);
            break;
        case IGATHER_BINOMIAL:
            res= gather_binomial(rank, p, root, schedule,
                                 sendbuf, sendcount, sendtype,
                                 recvbuf, recvcount, recvtype,
                                 inplace, &tmpbuf);
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
static inline int
gather_binomial(int rank, int p, int root, BULLNBC_Schedule *schedule,
                const void * sendbuf, int scount, MPI_Datatype sendtype,
                void * recvbuf, int rcount, MPI_Datatype recvtype,
                int inplace, void ** tmpbuf_ptr){
    char *tmpbuf;
    int res;
    size_t block_size; /* size per rank */
    int vrank = (rank - root + p) % p; /* Rankrotation to have vrank = 0 at root */
    MPI_Aint rextent; /* valid only for root */
    if (vrank != 0){
        rcount = scount;
        recvtype = sendtype;
    }

    opal_datatype_type_size(&recvtype->super, &block_size);
    block_size *= rcount;
    tmpbuf = *tmpbuf_ptr = (char *) malloc(block_size * p);
    if ( NULL == tmpbuf){
        NBC_Error("MPI Error in malloc()");
        return OMPI_ERR_OUT_OF_RESOURCE;
    }


    int mycount = 0; /* Number of block of data to send, including current rank */
    if (vrank == 0){
        ompi_datatype_type_extent(recvtype, &rextent);

        if (! inplace){
            /* Copy root data to recvbuf */
            res = NBC_Sched_copy ((void*)sendbuf, false, scount, sendtype,
                        (char*) recvbuf + rank * rextent * rcount, false, rcount, recvtype, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }
    } else { /* Leaf node send at least its data */
        mycount = 1;
    }

    uint32_t mask = 1;
    /* For each child, recv the data */
    while(vrank < (int) (vrank ^ mask)){

        int kid, kidcount, vkid;

        /* virtual id of the child */
        vkid = vrank ^ mask;
        if (vkid >= p){
            break; /* The end of the tree is reached */
        }
        /* From how many ranks this child has the data */
        kidcount = vkid - vrank;
        if ( kidcount > p - vkid){
            kidcount = p - vkid;
        }

        kid = (vkid + root ) % p;
        res = NBC_Sched_recv ((char*)tmpbuf + mycount * block_size,  false,
                              block_size * kidcount, MPI_BYTE, kid, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }

        mycount += kidcount;
        mask <<= 1;
    }

    if (vrank != 0){
        int vparent = vrank & (vrank - 1);
        int parent = (vparent + root) % p;

        if (mycount > 1){ /* not a leaf */
            /* Pack my data with children's one + wait all recv */
            res = NBC_Sched_copy ((void *)sendbuf, false, scount, sendtype,
                        tmpbuf, false, block_size, MPI_BYTE, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }

            res = NBC_Sched_send(tmpbuf, false, mycount * block_size, MPI_BYTE, parent, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        } else { /* leaf */
            res = NBC_Sched_send(sendbuf, false, scount, sendtype, parent, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }
    } else { /* root */
        /* Wait all recv */
        res = NBC_Sched_barrier(schedule);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }

        /* Undo rank rotation */
        res = NBC_Sched_copy (tmpbuf, false, (p - 1 - root) * block_size, MPI_BYTE,
                              (char*)recvbuf + rextent * (root + 1) * rcount, false,
                              (p - root - 1) * rcount, recvtype, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }
        res = NBC_Sched_copy (tmpbuf + (p - 1 - root) * block_size, false, root * block_size, MPI_BYTE,
                    recvbuf, false, root * rcount, recvtype, schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }

    }
    return MPI_SUCCESS;
}

static inline int
gather_linear(int rank, int p, int root, BULLNBC_Schedule *schedule,
              const void * sendbuf, int sendcount, MPI_Datatype sendtype,
              void * recvbuf, int recvcount, MPI_Datatype recvtype, int inplace){
    int res;


    /* send to root */
    if (rank != root) {
      /* send msg to root */
      res = NBC_Sched_send(sendbuf, false, sendcount, sendtype, root, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
      }
    } else {

      MPI_Aint ddtext;
      res = ompi_datatype_type_extent (sendtype, &ddtext);
      if (MPI_SUCCESS != res) {
          NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
          return res;
      }

      for (int i = 0 ; i < p ; ++i) {
        char * rbuf = (char *)recvbuf + i * recvcount * ddtext;
        if (i == root) {
          if (!inplace) {
            /* if I am the root - just copy the message */
            res = NBC_Sched_copy ((void *)sendbuf, false, sendcount, sendtype,
                                  rbuf, false, recvcount, recvtype, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
              return res;
            }
          }
        } else {
          /* root receives message to the right buffer */
          res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
          if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
          }
        }
      }
    }
    return res;
}

int ompi_coll_bullnbc_igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                             int recvcount, MPI_Datatype recvtype, int root,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
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

static int nbc_gather_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                  int recvcount, MPI_Datatype recvtype, int root,
                                  struct ompi_communicator_t *comm, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
    int res, rsize;
    MPI_Aint rcvext = 0;
    BULLNBC_Schedule *schedule;
    char *rbuf;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    rsize = ompi_comm_remote_size (comm);

    if (root == MPI_ROOT) {
        res = ompi_datatype_type_extent(recvtype, &rcvext);
        if (MPI_SUCCESS != res) {
          NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
          return res;
        }
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* send to root */
    if (root != MPI_ROOT && root != MPI_PROC_NULL) {
        /* send msg to root */
        res = NBC_Sched_send (sendbuf, false, sendcount, sendtype, root, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
          OBJ_RELEASE(schedule);
          return res;
        }
    } else if (MPI_ROOT == root) {
        for (int i = 0 ; i < rsize ; ++i) {
            rbuf = ((char *)recvbuf) + (i * recvcount * rcvext);
            /* root receives message to the right buffer */
            res = NBC_Sched_recv (rbuf, false, recvcount, recvtype, i, schedule, false);
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

int ompi_coll_bullnbc_igather_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                   int recvcount, MPI_Datatype recvtype, int root,
                                   struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
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

int ompi_coll_bullnbc_gather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                 int recvcount, MPI_Datatype recvtype, int root,
                                 struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gather_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                              comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_gather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                                       int recvcount, MPI_Datatype recvtype, int root,
                                       struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_gather_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                                    comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}


int
ompi_coll_bullnbc_pgatherr_linear (const void *sbuf, size_t sendparts, int sendcount,
                                   struct ompi_datatype_t *sdatatype,
                                   ompi_request_t ** sendreqs,
                                   void* rbuf, size_t recvparts, int recvcount,
                                   struct ompi_datatype_t *rdatatype,
                                   ompi_request_t ** recvreqs,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   const struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   const struct mca_coll_base_module_2_4_0_t *module)
{
    unsigned int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    unsigned int nrecv, nsend, nrreqs;

    if (rank != root) {
        rbuf = NULL;
        recvparts = 0;
        recvreqs = MPIX_NO_REQUESTS;
        nrecv = 0;
        nsend = 1;
        nrreqs = 0;
    } else {
        nrecv = comm_size -1;
        nsend = 0;
        nrreqs = comm_size * recvparts;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sendparts, sendreqs,
                                                   nrreqs, recvreqs,
                                                   nsend + nrecv, nrecv,
                                                   comm,  request);

    pcoll_dag_item** nodes = bullnbc_xmalloc((2*sendparts + nrreqs)
                                             * sizeof(pcoll_dag_item*));
    pcoll_dag_item ** ready = nodes;
    pcoll_dag_item ** scomplete = nodes + sendparts;
    pcoll_dag_item ** rcomplete = nodes + 2*sendparts;
    for (size_t part=0; part < sendparts; ++part){
        ready[part] = sched_ready(part, req, sendparts);
        scomplete[part] = sched_complete(part, req); /* send complete*/
    }

    if (rank == root) {
        ptrdiff_t extent;
        int ret = ompi_datatype_type_extent(rdatatype, &extent);
        if (MPI_SUCCESS != ret) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
            return ret;
        }

        /* complete recvreqs */
        for (size_t part=0; part < comm_size * recvparts; ++part){
            rcomplete[part] = sched_complete(sendparts + part, req);
        }
        pcoll_dag_item **root_rcomplete = rcomplete + root*recvparts;

        /* Handle root data transfer to recvuf */
        if (sbuf == MPI_IN_PLACE) {
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], scomplete[spart]);
                for (size_t rpart=0; rpart < recvparts; ++rpart){
                    create_dependency(ready[spart], root_rcomplete[rpart]);
                }
            }
        } else { /* Not in place */
            char * root_buf = ((char*)rbuf) + root*recvparts*recvcount*extent;
            pcoll_dag_item *move;
            move = sched_convertion(sbuf, sendparts * sendcount, sdatatype,
                                    root_buf, recvparts*recvcount, rdatatype,
                                    sendparts + recvparts);
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], move);
                create_dependency(move, scomplete[spart]);
            }
            for (size_t rpart=0; rpart < recvparts; ++rpart){
                create_dependency(move, root_rcomplete[rpart]);
            }
        }

        for (int peer=0; peer < comm_size; ++ peer){
            if (peer == rank) {
                continue; /* case done before the for loop */
            }
            char * child_rbuf = ((char*)rbuf) + peer*recvparts*recvcount*extent;
            pcoll_dag_item* recv;
            recv = sched_recv_v2(child_rbuf, recvcount*recvparts, rdatatype,
                                 peer, comm, req);

            pcoll_dag_item ** peer_rcomplete = rcomplete + peer*recvparts;
            for (size_t rpart=0; rpart < recvparts; ++rpart){
                create_dependency(recv, peer_rcomplete[rpart]);
            }
        }
    } else { /* Non root */
        pcoll_dag_item* send;
        send = sched_send_v2(sbuf, sendcount*sendparts, sdatatype,
                             root, comm, req);
        for (size_t spart=0; spart < sendparts; ++spart){
            create_dependency(ready[spart], send);
            create_dependency(send, scomplete[spart]);
        }
    }

    free(nodes);

    return OMPI_SUCCESS;
}

int
ompi_coll_bullnbc_pgatherr_partitioned_linear (const void *sbuf, size_t sendparts, int sendcount,
                                               struct ompi_datatype_t *sdatatype,
                                               ompi_request_t ** sendreqs,
                                               void* rbuf, size_t recvparts, int recvcount,
                                               struct ompi_datatype_t *rdatatype,
                                               ompi_request_t ** recvreqs,
                                               int root,
                                               struct ompi_communicator_t *comm,
                                               const struct ompi_info_t *info,
                                               ompi_request_t ** request,
                                               const struct mca_coll_base_module_2_4_0_t *module)
{
    int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    int nrecv, nsend, nrreqs;

    if (rank != root) {
        rbuf = NULL;
        recvparts = 0;
        recvreqs = MPIX_NO_REQUESTS;
        nrecv = 0;
        nsend = sendparts;
        nrreqs = 0;
    } else {
        nrecv = sendparts *(comm_size -1);
        nsend = 0;
        nrreqs = comm_size * recvparts;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sendparts, sendreqs,
                                                   nrreqs, recvreqs,
                                                   nsend + nrecv, nrecv,
                                                   comm,  request);
    ptrdiff_t sextent;
    int ret = ompi_datatype_type_extent(rdatatype, &sextent);
    if (MPI_SUCCESS != ret) {
        NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
        return ret;
    }

    for (size_t part=0; part < sendparts; ++part){

        if (rank == root) {

            ptrdiff_t rextent;
            ret = ompi_datatype_type_extent(rdatatype, &rextent);
            if (MPI_SUCCESS != ret) {
                NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
                return ret;
            }

            for (int peer=0; peer < comm_size; ++ peer){
                /* Assumption is made of uniform patition count,
                 * thus each sendpart matches a single recvpart */
                size_t complete_idx = sendparts + peer*recvparts + part;
                pcoll_dag_item * rcomplete = sched_complete(complete_idx, req);

                if (root == peer) {
                    pcoll_dag_item * ready = sched_ready(part, req, 1);
                    pcoll_dag_item * scomplete = sched_complete(part, req);
                    if (sbuf == MPI_IN_PLACE) {
                        create_dependency(ready, scomplete);
                        create_dependency(ready, rcomplete);
                    } else { /* Not in place */
                        pcoll_dag_item *move;
                        char* dest_buf = rbuf;
                        dest_buf += (root*recvparts + part) *recvcount*rextent;
                        const char* src_buf = (const char*)sbuf;
                        src_buf += part*sendcount*sextent;
                        move = sched_convertion(src_buf, sendcount, sdatatype,
                                                dest_buf, recvcount, rdatatype,
                                                2);
                        create_dependency(ready, move);
                        create_dependency(move, scomplete);
                        create_dependency(move, rcomplete);
                    }
                } else { /* Not my parts */
                    char* child_rbuf = rbuf;
                    child_rbuf += (peer*recvparts + part) *recvcount*rextent;
                    pcoll_dag_item* recv;
                    recv = sched_recv_v2(child_rbuf, recvcount, rdatatype,
                                         peer, comm, req);
                    create_dependency(recv, rcomplete);
                }
            }
        } else { /* Non root */
            pcoll_dag_item* send;
            pcoll_dag_item * ready = sched_ready(part, req, 1);
            pcoll_dag_item * scomplete = sched_complete(part, req);
            const char * src_buf = (const char*)sbuf;
            src_buf += part*sendcount*sextent;
            send = sched_send_v2(src_buf, sendcount, sdatatype, root, comm, req);
            create_dependency(ready, send);
            create_dependency(send, scomplete);
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
 * Each process gathers partitioned data from its direct children.
 * For a given rank R, let V=R-root the virtual rank, children rank are all V+mask
 * with mask any power of 2 lower than the least non zero significant bit.
 *
 * For non root and non leaf process the data from all the children must
 * be gathered before performing a single send.
 * For these processes, data gathering is made in a temporary buffer
 * ----------------------------------------------------------------------------
 * |   rank  | rank +b1 |      rank +b10       |                 rank +b100
 * ----------------------------------------------------------------------------
 *  For root rbuf is used as receive buffer. If root is not rank 0, a receive may
 *  cover both last ranks an first rank. In this case a derived datatype is created.
 *
 *  Note: to avoid a copy at the end, root rank receives data directly in the
 *  user receive buffer. That means that root shifts indexes of recvs to fit
 *  with real ranks whereas other ranks index recvs on virtual ranking
 *  (where root is always 0)
 */
int
ompi_coll_bullnbc_pgatherr_binomial (const void *sbuf, size_t sendparts, int sendcount,
                                     struct ompi_datatype_t *sdatatype,
                                     ompi_request_t ** sendreqs,
                                     void* rbuf, size_t recvparts, int recvcount,
                                     struct ompi_datatype_t *rdatatype,
                                     ompi_request_t ** recvreqs,
                                     int root,
                                     struct ompi_communicator_t *comm,
                                     const struct ompi_info_t *info,
                                     ompi_request_t ** request,
                                     const struct mca_coll_base_module_2_4_0_t *module)
{
    unsigned int comm_size, rank;
    comm_size = ompi_comm_size (comm);
    rank = ompi_comm_rank (comm);
    unsigned int vrank = (rank - root + comm_size) % comm_size;
    unsigned int nsend, nrreqs;

    bool inplace = (sbuf == MPI_IN_PLACE);
    if (inplace) {
        sdatatype = rdatatype;
        sendparts = recvparts;
        sendcount = recvcount;
    }

    int parent = -1;
    unsigned int nrecv = 0;

    const unsigned int max_log2 = 32;
    int vchild[max_log2];
    ptrdiff_t vchild_block_offset[max_log2];
    size_t nblock_child[max_log2];

    vchild_block_offset[0] = 1; /* Prepare room for my data */
    uint32_t power2 = 1;
    /* Compute the number of receives which is also the number of children
     * and the number of ending 0 in vrank binary writting */
    while ((vrank & power2) == 0) {
        vchild[nrecv] = vrank + power2;
        if (vchild[nrecv] >= comm_size) {
            break;
        }
        if (vchild[nrecv] + power2 > comm_size) {
            nblock_child[nrecv] = comm_size - vchild[nrecv];
        } else {
            nblock_child[nrecv] = 1 << nrecv;
        }

        vchild_block_offset[nrecv+1] = vchild_block_offset[nrecv] + nblock_child[nrecv];
        power2 <<=1;

        nrecv++;
    }

    if (vrank > 0) {
        int vparent;
        int hibit = 1<<nrecv;
        while (!(vrank & hibit)) {
        /* previous nrecv loop exited with the break conndition */
            hibit <<=1;
        }
        vparent = vrank - hibit;
        parent = (vparent + root) % comm_size;

        nsend = 1;
        nrreqs = 0;
        recvreqs = MPIX_NO_REQUESTS;

    } else {
        nsend = 0;
        nrreqs = comm_size * recvparts;
    }

    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_pcoll_init_dag_req(sendparts, sendreqs,
                                                   nrreqs, recvreqs,
                                                   nsend + nrecv, nrecv,
                                                   comm,  request);
    NBC_DEBUG(10, "Binomial gather %p, root %d\n", req, root);
    pcoll_dag_item** nodes = bullnbc_xmalloc((2*sendparts + nrreqs)
                                             * sizeof(pcoll_dag_item*));
    pcoll_dag_item ** ready = nodes;
    pcoll_dag_item ** scomplete = nodes + sendparts;
    pcoll_dag_item ** rcomplete = nodes + 2*sendparts;
    for (size_t part=0; part < sendparts; ++part){
        ready[part] = sched_ready(part, req, sendparts);
        scomplete[part] = sched_complete(part, req); /* send complete*/
    }

    if (rank == root) {
        /* complete recvreqs */
        for (size_t part=0; part < comm_size * recvparts; ++part){
            rcomplete[part] = sched_complete(sendparts + part, req);
        }

        ptrdiff_t rextent;
        int ret = ompi_datatype_type_extent(rdatatype, &rextent);
        if (MPI_SUCCESS != ret) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
            return ret;
        }

        size_t block_count = recvcount*recvparts;
        size_t block_extent = rextent*block_count;
        char * root_buf = ((char*) rbuf) + root*block_extent;

        /* Handle root data transfer to recvuf */
        pcoll_dag_item **root_rcomplete = rcomplete + root*recvparts;
        if (sbuf == MPI_IN_PLACE) {
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], scomplete[spart]);
                for (size_t rpart=0; rpart < recvparts; ++rpart){
                    create_dependency(ready[spart], root_rcomplete[rpart]);
                }
            }
        } else { /* Not in place */
            pcoll_dag_item *move;
            move = sched_convertion(sbuf, sendparts * sendcount, sdatatype,
                                    root_buf, recvparts*recvcount, rdatatype,
                                    sendparts + recvparts);
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], move);
                create_dependency(move, scomplete[spart]);
            }
            for (size_t rpart=0; rpart < recvparts; ++rpart){
                create_dependency(move, root_rcomplete[rpart]);
            }
        }

        /* Don't care of the dependencies order, since it never
         * unlocks more than 1 comm */
        for (unsigned int child_idx = 0; child_idx < nrecv; ++child_idx) {
            int child = (vchild[child_idx] + root) % comm_size;

            char * child_rbuf;
            size_t child_rcount = nblock_child[child_idx] * block_count;
            /* This variable stores the received dataype. It may not be rdatatype
             * in case a derived datatype is needed */
            struct ompi_datatype_t * child_rddt = rdatatype;
            size_t block_ranges[2][2] = {{0}};
            size_t nblock_ranges = 1;
            block_ranges[0][0] = child;
            block_ranges[0][1] = child + nblock_child[child_idx];

            if (root + vchild_block_offset[child_idx] >= comm_size) {
                /* Root is not 0 so last vchildren are in fact first ranks.
                 * Apply a modulo to put the data at the buffer start */
                ptrdiff_t block_start = root + vchild_block_offset[child_idx];
                block_start %= comm_size;
                child_rbuf = ((char*) rbuf) + block_start * block_extent;

            } else if (root + vchild_block_offset[child_idx+1] > comm_size) {
                /* Ends out of bounds which means recv last and first ranks data */
                nblock_ranges = 2;
                block_ranges[1][0] = 0;
                block_ranges[1][1] = block_ranges[0][1] - comm_size;
                block_ranges[0][1] = comm_size;

                unsigned int split_len[2];
                unsigned int split_disp[2];
                size_t nblock_after = block_ranges[0][1] - block_ranges[0][0];
                size_t nblock_before = block_ranges[1][1] - block_ranges[1][0];
                split_len[0] = nblock_after*block_count;
                split_len[1] = nblock_before*block_count;
                split_disp[0] = child * block_count;
                split_disp[1] = 0;
                ompi_datatype_create_indexed(2, split_len, split_disp,
                                             rdatatype, &child_rddt);
                ompi_datatype_commit(&child_rddt);

                req->n_created_ddt = 1;
                req->created_ddt = bullnbc_xmalloc(sizeof(struct ompi_datatype_t *));
                req->created_ddt[0] = child_rddt;

                child_rbuf = rbuf;
                child_rcount = 1;
                NBC_DEBUG(10, "recv blocks [%d-%d[+[%d-%d[ (%dx%d ddt) from %d(vr:%d) \n",
                      block_ranges[0][0], block_ranges[0][1],
                      block_ranges[1][0], block_ranges[1][1],
                      nblock_child[child_idx], block_count, child, vchild[child_idx]);
            } else {
                child_rbuf = root_buf + vchild_block_offset[child_idx] * block_extent;
            }


            pcoll_dag_item* recv;
            recv = sched_recv_v2(child_rbuf, child_rcount, child_rddt,
                                 child, comm, req);

            for (size_t range=0; range < nblock_ranges; ++range){
                size_t start = block_ranges[range][0] * recvparts;
                size_t end = block_ranges[range][1] * recvparts;

                for (size_t rpart=start; rpart < end; ++rpart){
                    create_dependency(recv, rcomplete[rpart]);
                }
            }
        }

    } else { /* Non root */

        ptrdiff_t sextent;
        int ret = ompi_datatype_type_extent(sdatatype, &sextent);
        if (MPI_SUCCESS != ret) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", ret);
            return ret;
        }
        size_t block_count = sendcount*sendparts;
        size_t block_extent = sextent*block_count;

        void * send_tmpbuf;
        if (nrecv > 0) { /* intermediary ranks */
            size_t tmpbuf_size = vchild_block_offset[nrecv];
            tmpbuf_size *= sextent* sendparts *sendcount;
            req->tmpbuf = bullnbc_xmalloc(tmpbuf_size);
            send_tmpbuf = req->tmpbuf;
        } else {
            send_tmpbuf = (void*) sbuf;
        }
        pcoll_dag_item* send;
        size_t nblock_send  = vchild_block_offset[nrecv] * sendcount * sendparts;
        NBC_DEBUG(10, "Send %d ddt to %d \n", nblock_send, parent);
        send = sched_send(send_tmpbuf, nblock_send, sdatatype,
                             parent, comm, NULL, req, sendparts);

        if (nrecv > 0) { /* intermediary ranks */
            pcoll_dag_item *move;
            move = sched_convertion(sbuf, sendparts * sendcount, sdatatype,
                                    send_tmpbuf, sendparts * sendcount, sdatatype,
                                    sendparts +1);
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], move);
                create_dependency(move, scomplete[spart]);
            }
            create_dependency(move, send);


            /* Don't care of the dependencies order, since it never
             * unlocks more than 1 comm */
            for (unsigned int child_idx = 0; child_idx < nrecv; ++child_idx) {
                int child = (vchild[child_idx] + root) % comm_size;
                char * child_rbuf = req->tmpbuf;
                child_rbuf += vchild_block_offset[child_idx] * block_extent;
                size_t child_rcount = nblock_child[child_idx] * block_count;

                pcoll_dag_item* recv;
                recv = sched_recv_v2(child_rbuf, child_rcount, sdatatype,
                                     child, comm, req);
                create_dependency(recv, send);
            }

        } else { /* leaves */
            for (size_t spart=0; spart < sendparts; ++spart){
                create_dependency(ready[spart], send);
                create_dependency(send, scomplete[spart]);
            }
        }
    }

    free(nodes);

    return OMPI_SUCCESS;
}


int
ompi_coll_bullnbc_pgatherr_init (const void *sbuf, size_t sendparts, int sendcount,
                                 struct ompi_datatype_t *sdatatype,
                                 ompi_request_t ** sendreqs,
                                 void* rbuf, size_t recvparts, int recvcount,
                                 struct ompi_datatype_t *rdatatype,
                                 ompi_request_t ** recvreqs,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 struct ompi_info_t *info,
                                 ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module)
{
    int comm_size = ompi_comm_size (comm);
    size_t total_size;
    ompi_datatype_type_size(sdatatype, &total_size);
    total_size *= sendcount*sendparts;

    if ((sbuf == MPI_IN_PLACE && comm_size < 2)
        || 0 == total_size) {
        NBC_DEBUG(5, "This MPI_Pgatherr_init is a noop : %d peers on %d bytes from %p\n",
                  comm_size, total_size, sbuf);
        ompi_mca_coll_bullnbc_pcoll_init_empty_req(sendparts, sendreqs,
                                                   recvparts, recvreqs,
                                                   comm, request);
        return MPI_SUCCESS;
    }

    int alg = pgatherr_algorithm;

    if (comm_size < 2) {
        alg = 1;    /* Just a copy to rbuf: done by linear */
    }

    if (0 == alg) {
        alg = 1;
    }

    if (! mca_coll_bullnbc_uniform_partition_sizes && 3 == alg){
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Non uniform partition sizes", true,
                       3, "gather", 1);
        alg = 1;
    }

    module_need_progress((ompi_coll_bullnbc_module_t*) module);

    NBC_DEBUG(2, "Use algorithm %d for pgatherr with %d sendparts,sendcount %d and root %d\n",
              alg, sendparts, sendcount, root);
    switch (alg){
        case 1:
            return ompi_coll_bullnbc_pgatherr_linear(sbuf, sendparts, sendcount,
                                                     sdatatype, sendreqs,
                                                     rbuf, recvparts, recvcount, rdatatype,
                                                     recvreqs,
                                                     root, comm, info,
                                                     request, module);
        case 2:
            return ompi_coll_bullnbc_pgatherr_binomial(sbuf, sendparts, sendcount,
                                                     sdatatype, sendreqs,
                                                     rbuf, recvparts, recvcount, rdatatype,
                                                     recvreqs,
                                                     root, comm, info,
                                                     request, module);
        case 3:
            return ompi_coll_bullnbc_pgatherr_partitioned_linear(sbuf, sendparts, sendcount,
                                                                 sdatatype, sendreqs,
                                                                 rbuf, recvparts, recvcount, rdatatype,
                                                                 recvreqs,
                                                                 root, comm, info,
                                                                 request, module);
        default:
            fprintf(stderr, "Pgatherr algorithm %d is not implemented yet\n", alg); fflush(stderr);
            abort();
    }
}
int
ompi_coll_bullnbc_pgather_init (const void *sbuf, size_t sendparts, int sendcount,
                                struct ompi_datatype_t *sdatatype,
                                void* rbuf, size_t recvparts, int recvcount,
                                struct ompi_datatype_t *rdatatype,
                                int root,
                                struct ompi_communicator_t *comm,
                                struct ompi_info_t *info,
                                ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module)
{
    return ompi_coll_bullnbc_pgatherr_init(sbuf, sendparts, sendcount,
                                           sdatatype, MPIX_NO_REQUESTS,
                                           rbuf, recvparts, recvcount,
                                           rdatatype, MPIX_NO_REQUESTS,
                                           root, comm, info, request, module);
}
