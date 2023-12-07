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

static  inline int
nbc_scatter_linear (int rank, int p, int root, BULLNBC_Schedule *schedule, const void * sendbuf, int scount, MPI_Datatype sendtype, void * recvbuf, int rcount, MPI_Datatype recvtype, int inplace);

static  inline int
nbc_scatter_binomial (int rank, int p, int root, BULLNBC_Schedule *schedule, const void * sendbuf, int scount, MPI_Datatype sendtype, void * recvbuf, int rcount, MPI_Datatype recvtype, int inplace, void ** tmpbuf_ptr);

static mca_base_var_enum_value_t scatter_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "binomial"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_scatter_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != scatter_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iscatter_algorithm_count",
                                         "Number of scatter algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_scatter_algorithms", scatter_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iscatter_algorithm",
                                         "Which scatter algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "iscatter_algorithm_segmentsize",
                                  "Segment size in bytes used by default for iscatter algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

/* simple linear MPI_Iscatter */
static int nbc_scatter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                             void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
  size_t size;
  int rank, p, res;
  BULLNBC_Schedule *schedule;
  char inplace = 0;
  void *tmpbuf = NULL;
  enum {ISCATTER_LINEAR, ISCATTER_BINOMIAL} alg;
  ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;


  rank = ompi_comm_rank (comm);
  if (root == rank) {
    NBC_IN_PLACE(sendbuf, recvbuf, inplace);
  }
  p = ompi_comm_size (comm);

  res = ompi_datatype_type_size (recvtype, &size);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
      return res;
  }
  size *= recvcount;

  if(mca_coll_bullnbc_component.use_dynamic_rules) {
      if(0 != mca_coll_bullnbc_component.forced_params[SCATTER].algorithm) {
          /* if op is not commutative or MPI_IN_PLACE was specified we have to deal with it */
          alg = mca_coll_bullnbc_component.forced_params[SCATTER].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
          goto selected_rule;
      }
      if(bullnbc_module->com_rules[SCATTER]) {
          int algorithm,dummy1,dummy2,dummy3;
          algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[SCATTER],
                      size, &dummy1, &dummy2, &dummy3);
          if(algorithm) {
              alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
              goto selected_rule;
          }
      }
  }
  if (size < 256){
      alg = ISCATTER_BINOMIAL;
  } else {
      alg = ISCATTER_LINEAR;
  }

selected_rule:
  opal_output_verbose(10, mca_coll_bullnbc_component.stream,
                      "Bullnbc iscatter : algorithm %d (no segmentation supported)",
                      alg + 1);

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
      return OMPI_ERR_OUT_OF_RESOURCE;
    }

    switch(alg){
        case ISCATTER_LINEAR:
            res = nbc_scatter_linear(rank, p, root, schedule,
                                     sendbuf, sendcount, sendtype,
                                     recvbuf, recvcount, recvtype, inplace);
        break;
        case  ISCATTER_BINOMIAL:
            res = nbc_scatter_binomial(rank, p, root, schedule,
                                       sendbuf, sendcount, sendtype,
                                       recvbuf, recvcount, recvtype, inplace, &tmpbuf);
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
static  inline int
nbc_scatter_linear (int rank, int p, int root, BULLNBC_Schedule *schedule, const void * sendbuf, int scount, MPI_Datatype sendtype, void * recvbuf, int rcount, MPI_Datatype recvtype, int inplace){
    int res;

    /* receive from root */
    if (rank != root) {
      /* recv msg from root */
      res = NBC_Sched_recv (recvbuf, false, rcount, recvtype, root, schedule, false);
      if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        OBJ_RELEASE(schedule);
        return res;
      }
    } else {
        MPI_Aint sndext = 0;
        res = ompi_datatype_type_extent (sendtype, &sndext);
        if (MPI_SUCCESS != res) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
            return res;
        }
        for (int i = 0 ; i < p ; ++i) {
        char * sbuf;
        sbuf = (char *) sendbuf + i * scount * sndext;
        if (i == root) {
          if (!inplace) {
            /* if I am the root - just copy the message */
            res = NBC_Sched_copy (sbuf, false, scount, sendtype,
                                  recvbuf, false, rcount, recvtype, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
              OBJ_RELEASE(schedule);
              return res;
            }
          }
        } else {
          /* root sends the right buffer to the right receiver */
          res = NBC_Sched_send (sbuf, false, scount, sendtype, i, schedule, false);
          if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
          }
        }
      }
    }
    return res;
}

static  inline int
nbc_scatter_binomial (int rank, int p, int root, BULLNBC_Schedule *schedule,
                      const void * sendbuf, int scount, MPI_Datatype sendtype,
                      void * recvbuf, int rcount, MPI_Datatype recvtype,
                      int inplace, void ** tmpbuf_ptr){
    int vrank;
    ptrdiff_t sextent;

    char *tmpbuf;
    int res, is_leaf;
    size_t block_size; /* size per rank */

    vrank = (rank - root + p) % p;

    opal_datatype_type_size(&recvtype->super, &block_size);
    block_size *= rcount;

    is_leaf = (vrank & 1) || (vrank == p - 1);
    if (!is_leaf){
        /* Allocate an buffer filled by the parent for children data */
        tmpbuf = *tmpbuf_ptr = (char *) malloc(block_size * p);
        if ( NULL == tmpbuf){
            NBC_Error("MPI Error in malloc()");
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
    }

    if (vrank == 0) {
        ompi_datatype_type_extent(sendtype, &sextent);

        if (! inplace) {
            res = NBC_Sched_copy ((char*)sendbuf + root * scount * sextent, false, scount, sendtype,
                        recvbuf, false, rcount, recvtype, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }

        /* Data shift in send buffer: shifted by root times the block size */
        res = NBC_Sched_copy ((char*)sendbuf, false, root * scount, sendtype,
                              tmpbuf +  (p-1-root) * block_size, false,
                              root * block_size, MPI_BYTE, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }
        res = NBC_Sched_copy ((char*)sendbuf + ( root + 1 ) * sextent * scount, false, (p - root - 1) * scount , sendtype,
                              tmpbuf, false, (p - 1 - root) * block_size,  MPI_BYTE,
                              schedule, true);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }
    } else {
        int vparent = vrank & (vrank - 1);
        int parent = (vparent + root) % p;
        int mycount = vrank - vparent;
        if (mycount + vrank >= p){
            mycount = p - vrank;
        }
        /* Recv my data in recvbuf */
        res = NBC_Sched_recv (recvbuf, false, rcount, recvtype, parent, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }

        if (mycount > 1){ /* Not leaf */
            /* Recv children data in tmpbuf */
            res = NBC_Sched_recv ((char*)tmpbuf,  false, (mycount - 1) * block_size, MPI_BYTE, parent, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
        }
    }

    uint32_t mask = 1;
    ptrdiff_t offset = 0;
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
        res = NBC_Sched_send(tmpbuf + offset, false, block_size, MPI_BYTE, kid, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            return res;
        }
        offset += block_size;

        if ( kidcount > 1) { /* Kid is not a leaf */
            res = NBC_Sched_send(tmpbuf + offset , false, (kidcount -1) * block_size, MPI_BYTE, kid, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                return res;
            }
            offset += (kidcount - 1) * block_size;
        }

        mask <<= 1;
    }

    return res;
}

int ompi_coll_bullnbc_iscatter (const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                               void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
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

static int nbc_scatter_inter_init (const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                   void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                   struct ompi_communicator_t *comm, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
    int res, rsize;
    MPI_Aint sndext;
    BULLNBC_Schedule *schedule;
    char *sbuf;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;

    rsize = ompi_comm_remote_size (comm);

    if (MPI_ROOT == root) {
        res = ompi_datatype_type_extent(sendtype, &sndext);
        if (MPI_SUCCESS != res) {
            NBC_Error("MPI Error in ompi_datatype_type_extent() (%i)", res);
            return res;
        }
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    /* receive from root */
    if (MPI_ROOT != root && MPI_PROC_NULL != root) {
        /* recv msg from remote root */
        res = NBC_Sched_recv(recvbuf, false, recvcount, recvtype, root, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            OBJ_RELEASE(schedule);
            return res;
        }
    } else if (MPI_ROOT == root) {
        for (int i = 0 ; i < rsize ; ++i) {
            sbuf = ((char *)sendbuf) + (i * sendcount * sndext);
            /* root sends the right buffer to the right receiver */
            res = NBC_Sched_send(sbuf, false, sendcount, sendtype, i, schedule, false);
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

int ompi_coll_bullnbc_iscatter_inter (const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                     void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                     struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatter_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
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

int ompi_coll_bullnbc_scatter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                  struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                               comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}

int ompi_coll_bullnbc_scatter_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                        void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                        struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_scatter_inter_init(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root,
                                     comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
