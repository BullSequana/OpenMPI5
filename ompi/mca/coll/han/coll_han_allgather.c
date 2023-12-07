/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2022      IBM Corporation. All rights reserved
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * This files contains all the hierarchical implementations of allgather
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

static int mca_coll_han_allgather_lb_task(void *task_args);
static int mca_coll_han_allgather_lg_task(void *task_args);
static int mca_coll_han_allgather_uag_task(void *task_args);
void
mca_coll_han_reorder_allgather_up(const void *sbuf,
                                  void *rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_communicator_t *low_comm,
                                  struct ompi_communicator_t *up_comm,
                                  mca_coll_han_module_t *han_module);
void
mca_coll_han_reorder_allgather_low(const void *sbuf,
                                   void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_communicator_t *low_comm,
                                   struct ompi_communicator_t *up_comm,
                                   mca_coll_han_module_t *han_module);

static inline void
mca_coll_han_set_allgather_args(mca_coll_han_allgather_t * args,
                                mca_coll_task_t * cur_task,
                                void *sbuf,
                                void *sbuf_inter_free,
                                int scount,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf,
                                int rcount,
                                struct ompi_datatype_t *rdtype,
                                int root_low_rank,
                                struct ompi_communicator_t *up_comm,
                                struct ompi_communicator_t *low_comm,
                                int w_rank,
                                bool noop,
                                bool is_mapbycore,
                                int *topo,
                                ompi_request_t * req)
{
    args->cur_task = cur_task;
    args->sbuf = sbuf;
    args->sbuf_inter_free = sbuf_inter_free;
    args->scount = scount;
    args->sdtype = sdtype;
    args->rbuf = rbuf;
    args->rcount = rcount;
    args->rdtype = rdtype;
    args->root_low_rank = root_low_rank;
    args->up_comm = up_comm;
    args->low_comm = low_comm;
    args->w_rank = w_rank;
    args->noop = noop;
    args->is_mapbycore = is_mapbycore;
    args->topo = topo;
    args->req = req;
}


/**
 * Main function for taskified allgather: calls lg task, a gather on low comm
 */
int
mca_coll_han_allgather_intra(const void *sbuf, int scount,
                             struct ompi_datatype_t *sdtype,
                             void *rbuf, int rcount,
                             struct ompi_datatype_t *rdtype,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t * module)
{
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                              comm, han_module->previous_allgather_module);
    }
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int low_rank = ompi_comm_rank(low_comm);
    int w_rank = ompi_comm_rank(comm);

    /* Init topo */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);
    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather with this communicator (imbalance). Fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return han_module->previous_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                              comm, han_module->previous_allgather_module);
    }

    ompi_request_t *temp_request;
    /* Set up request */
    temp_request = OBJ_NEW(ompi_request_t);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = OMPI_REQUEST_COLL;
    temp_request->req_free = ompi_coll_han_request_free;
    temp_request->req_status = (ompi_status_public_t){0};
    temp_request->req_complete = REQUEST_PENDING;

    int root_low_rank = 0;
    /* Create lg (lower level gather) task */
    mca_coll_task_t *lg = OBJ_NEW(mca_coll_task_t);
    /* Setup lg task arguments */
    mca_coll_han_allgather_t *lg_args = malloc(sizeof(mca_coll_han_allgather_t));
    mca_coll_han_set_allgather_args(lg_args, lg, (char *) sbuf, NULL, scount, sdtype, rbuf, rcount,
                                    rdtype, root_low_rank, up_comm, low_comm, w_rank,
                                    low_rank != root_low_rank, han_module->is_mapbycore, topo,
                                    temp_request);
    /* Init and issue lg task */
    init_task(lg, mca_coll_han_allgather_lg_task, (void *) (lg_args));
    issue_task(lg);

    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);

    return OMPI_SUCCESS;
}

/* lg: lower level gather task */
int mca_coll_han_allgather_lg_task(void *task_args)
{
    mca_coll_han_allgather_t *t = (mca_coll_han_allgather_t *) task_args;
    char *tmp_buf = NULL, *tmp_rbuf = NULL;
    char *tmp_send = NULL;

    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] HAN Allgather:  lg\n",
                         t->w_rank));

    /* If the process is one of the node leader */
    ptrdiff_t rlb, rext;
    ompi_datatype_get_extent (t->rdtype, &rlb, &rext);
    if (MPI_IN_PLACE == t->sbuf) {
        t->sdtype = t->rdtype;
        t->scount = t->rcount;
    }
    if (!t->noop) {
        int low_size = ompi_comm_size(t->low_comm);
        ptrdiff_t rsize, rgap = 0;
        rsize = opal_datatype_span(&t->rdtype->super, (int64_t) t->rcount * low_size, &rgap);
        tmp_buf = (char *) malloc(rsize);
        tmp_rbuf = tmp_buf - rgap;
        if (MPI_IN_PLACE == t->sbuf) {
            tmp_send = ((char*)t->rbuf) + (ptrdiff_t)t->w_rank * (ptrdiff_t)t->rcount * rext;
            ompi_datatype_copy_content_same_ddt(t->rdtype, t->rcount, tmp_rbuf, tmp_send);
        }
    }
    /* Lower level (shared memory or intra-node) gather */
    if (MPI_IN_PLACE == t->sbuf) {
        if (!t->noop) {
            t->low_comm->c_coll->coll_gather(MPI_IN_PLACE, t->scount, t->sdtype, 
                                             tmp_rbuf, t->rcount, t->rdtype, t->root_low_rank, 
                                             t->low_comm, t->low_comm->c_coll->coll_gather_module);
        }
        else {
            tmp_send = ((char*)t->rbuf) + (ptrdiff_t)t->w_rank * (ptrdiff_t)t->rcount * rext;
            t->low_comm->c_coll->coll_gather(tmp_send, t->rcount, t->rdtype, 
                                             NULL, t->rcount, t->rdtype, t->root_low_rank, 
                                             t->low_comm, t->low_comm->c_coll->coll_gather_module);
        }
    }
    else {
        t->low_comm->c_coll->coll_gather((char *) t->sbuf, t->scount, t->sdtype, tmp_rbuf, t->rcount,
                                         t->rdtype, t->root_low_rank, t->low_comm,
                                         t->low_comm->c_coll->coll_gather_module);
    }

    t->sbuf = tmp_rbuf;
    t->sbuf_inter_free = tmp_buf;

    /* Create uag (upper level all-gather) task */
    mca_coll_task_t *uag = t->cur_task;
    /* Init and issue uag task */
    init_task(uag, mca_coll_han_allgather_uag_task, (void *) t);
    issue_task(uag);

    return OMPI_SUCCESS;
}

/* uag: upper level (inter-node) all-gather task */
int mca_coll_han_allgather_uag_task(void *task_args)
{
    mca_coll_han_allgather_t *t = (mca_coll_han_allgather_t *) task_args;

    if (t->noop) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] HAN Allgather:  uag noop\n", t->w_rank));
    } else {
        int low_size = ompi_comm_size(t->low_comm);
        int up_size = ompi_comm_size(t->up_comm);
        char *reorder_buf = NULL;
        char *reorder_rbuf = NULL;
        if (t->is_mapbycore) {
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                 "[%d]: HAN Allgather is bycore: ", t->w_rank));
            reorder_rbuf = (char *) t->rbuf;
        } else {
            ptrdiff_t rsize, rgap = 0;
            rsize =
                opal_datatype_span(&t->rdtype->super,
                                   (int64_t) t->rcount * low_size * up_size,
                                   &rgap);
            reorder_buf = (char *) malloc(rsize);
            reorder_rbuf = reorder_buf - rgap;
        }

        /* Inter node allgather */
        t->up_comm->c_coll->coll_allgather((char *) t->sbuf, t->scount * low_size, t->sdtype,
                                           reorder_rbuf, t->rcount * low_size, t->rdtype,
                                           t->up_comm, t->up_comm->c_coll->coll_allgather_module);

        if (t->sbuf_inter_free != NULL) {
            free(t->sbuf_inter_free);
            t->sbuf_inter_free = NULL;
        }

        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] HAN Allgather:  ug allgather finish\n", t->w_rank));

        /* Reorder the node leader's rbuf, copy data from tmp_rbuf to rbuf */
        if (!t->is_mapbycore) {
            int i, j;
            ptrdiff_t rextent;
            ompi_datatype_type_extent(t->rdtype, &rextent);
            for (i = 0; i < up_size; i++) {
                for (j = 0; j < low_size; j++) {
                    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                         "[%d]: HAN Allgather copy from %d %d\n", t->w_rank,
                                         (i * low_size + j) * 2 + 1,
                                         t->topo[(i * low_size + j) * 2 + 1]));
                    ompi_datatype_copy_content_same_ddt(t->rdtype,
                                                        (ptrdiff_t) t->rcount,
                                                        (char *) t->rbuf +
                                                        rextent *
                                                        (ptrdiff_t) t->topo[(i * low_size + j) * 2 +
                                                                            1] *
                                                        (ptrdiff_t) t->rcount,
                                                        reorder_rbuf + rextent * (i * low_size +
                                                                                  j) *
                                                        (ptrdiff_t) t->rcount);
                }
            }
            free(reorder_buf);
            reorder_buf = NULL;
        }
    }


    /* Create lb (low level broadcast) task */
    mca_coll_task_t *lb = t->cur_task;
    /* Init and issue lb task */
    init_task(lb, mca_coll_han_allgather_lb_task, (void *) t);
    issue_task(lb);

    return OMPI_SUCCESS;
}

/* lb: low level broadcast task */
int mca_coll_han_allgather_lb_task(void *task_args)
{
    mca_coll_han_allgather_t *t = (mca_coll_han_allgather_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] HAN Allgather:  uag noop\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);
    int low_size = ompi_comm_size(t->low_comm);
    int up_size = ompi_comm_size(t->up_comm);
    t->low_comm->c_coll->coll_bcast((char *) t->rbuf, t->rcount * low_size * up_size, t->rdtype,
                                    t->root_low_rank, t->low_comm,
                                    t->low_comm->c_coll->coll_bcast_module);

    ompi_request_t *temp_req = t->req;
    free(t);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;

}

/**
 * Short implementation of allgather that only does hierarchical
 * communications without tasks.
 */
int
mca_coll_han_allgather_intra_simple(const void *sbuf, int scount,
                                    struct ompi_datatype_t *sdtype,
                                    void* rbuf, int rcount,
                                    struct ompi_datatype_t *rdtype,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module){

    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                              comm, han_module->previous_allgather_module);
    }
    /* discovery topology */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return han_module->previous_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                              comm, han_module->previous_allgather_module);
    }

    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int w_rank = ompi_comm_rank(comm);
    /* setup up/low coordinates */
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int up_rank = ompi_comm_rank(up_comm);
    int up_size = ompi_comm_size(up_comm);
    int root_low_rank = 0; // node leader will be 0 on each rank

    /* allocate the intermediary buffer
     * to gather on leaders on the low sub communicator */
    ptrdiff_t rlb, rext;
    ompi_datatype_get_extent (rdtype, &rlb, &rext);
    char *tmp_buf = NULL;
    char *tmp_buf_start = NULL;
    char *tmp_send = NULL;
    if (MPI_IN_PLACE == sbuf) {
        scount = rcount;
        sdtype = rdtype;
    }
    if (low_rank == root_low_rank) {
        ptrdiff_t rsize, rgap = 0;
        /* Compute the size to receive all the local data, including datatypes empty gaps */
        rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * low_size, &rgap);
        /* intermediary buffer on node leaders to gather on low comm */
        tmp_buf = (char *) malloc(rsize);
        tmp_buf_start = tmp_buf - rgap;
        if (MPI_IN_PLACE == sbuf) {
            tmp_send = ((char*)rbuf) + (ptrdiff_t)w_rank * (ptrdiff_t)rcount * rext;
            ompi_datatype_copy_content_same_ddt(rdtype, rcount, tmp_buf_start, tmp_send);
        }
    }

    /* 1. low gather on node leaders into tmp_buf */
    if (MPI_IN_PLACE == sbuf) {
        if (low_rank == root_low_rank) {
            low_comm->c_coll->coll_gather(MPI_IN_PLACE, scount, sdtype,
                                          tmp_buf_start, rcount, rdtype, root_low_rank,
                                          low_comm, low_comm->c_coll->coll_gather_module);
        }
        else {
            tmp_send = ((char*)rbuf) + (ptrdiff_t)w_rank * (ptrdiff_t)rcount * rext;
            low_comm->c_coll->coll_gather(tmp_send, rcount, rdtype,
                                          NULL, rcount, rdtype, root_low_rank,
                                          low_comm, low_comm->c_coll->coll_gather_module);
        }
    }
    else {
        low_comm->c_coll->coll_gather((char *)sbuf, scount, sdtype,
                                      tmp_buf_start, rcount, rdtype, root_low_rank,
                                      low_comm, low_comm->c_coll->coll_gather_module);
    }
    /* 2. allgather between node leaders, from tmp_buf to reorder_buf */
    if (low_rank == root_low_rank) {
        /* allocate buffer to store unordered result on node leaders
         * if the processes are mapped-by core, no need to reorder:
         * distribution of ranks on core first and node next,
         * in a increasing order for both patterns.
         */
        char *reorder_buf = NULL;
        char *reorder_buf_start = NULL;
        if (han_module->is_mapbycore) {
            reorder_buf_start = rbuf;
        } else {
            if (0 == low_rank && 0 == up_rank) { // first rank displays message
                OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                     "[%d]: Future Allgather needs reordering: ", up_rank));
            }
            ptrdiff_t rsize, rgap = 0;
            rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * low_size * up_size, &rgap);
            reorder_buf = (char *) malloc(rsize);
            reorder_buf_start = reorder_buf - rgap;
        }

        /* 2a. inter node allgather */
        up_comm->c_coll->coll_allgather(tmp_buf_start, scount*low_size, sdtype,
                                        reorder_buf_start, rcount*low_size, rdtype,
                                        up_comm, up_comm->c_coll->coll_allgather_module);

        if (tmp_buf != NULL) {
            free(tmp_buf);
            tmp_buf = NULL;
            tmp_buf_start = NULL;
        }

        /* 2b. reorder the node leader's into rbuf.
         * if ranks are not mapped in topological order, data needs to be reordered
         * (see reorder_gather)
         */
        if (!han_module->is_mapbycore) {
            ompi_coll_han_reorder_gather(reorder_buf_start,
                                         rbuf, rcount, rdtype,
                                         comm, topo);
            free(reorder_buf);
            reorder_buf = NULL;
        }

    }

    /* 3. up broadcast: leaders broadcast on their nodes */
    low_comm->c_coll->coll_bcast(rbuf, rcount*low_size*up_size, rdtype,
                                 root_low_rank, low_comm,
                                 low_comm->c_coll->coll_bcast_module);


    return OMPI_SUCCESS;
}

/* Allgather Low algorithm:
 * This algorithm is split in two steps:
 *
 * First an allgather is done on all low communicators
 * Then another allgather on each up communicator
 *
 * For a block distribution, no reorder is needed
 *
 * Example with reordering (Cyclic distribution):
 *
 * wrank(up_rank, low_rank)[Send_buf]
 *
 * 0 (0, 0)[A]    1 (1, 0)[B]
 *
 * 2 (0, 1)[C]    3 (1, 1)[D]
 *
 * 1) Allgather on low communicator [Tmp_buf]
 *
 * 0[A, C]        1[B, D]
 *
 * 2[A, C]        3[B, D]
 *
 * 2) Allgather on up communicator  [Reorder_buf]
 *
 * 0[A, C, B, D]        1[A, C, B, D]
 *
 * 2[A, C, B, D]        3[A, C, B, D]
 *
 * 3) Reorder on all [Reorder_buf] => [Recv_buf]
 *
 *  [A, C, B, D] => [A, B, C, D]
 *
 */
int
mca_coll_han_allgather_intra_low(const void *sbuf, int scount,
                                 struct ompi_datatype_t *sdtype,
                                 void* rbuf, int rcount,
                                 struct ompi_datatype_t *rdtype,
                                 struct ompi_communicator_t *comm,
                                 mca_coll_base_module_t *module)
{
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }
    /* discovery topology */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). Fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* setup up/low size */
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);

    /* Alloc Temporary recv buffer */
    char *tmp_recv_buf = NULL;
    char *tmp_recv_buf_start = NULL;
    ptrdiff_t rsize, rgap = 0;
    rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * low_size , &rgap);
    tmp_recv_buf = (char *) malloc(rsize);
    tmp_recv_buf_start = tmp_recv_buf - rgap;

    /* MPI IN PLACE */
    if (MPI_IN_PLACE == sbuf) {
        ptrdiff_t rextent;
        ompi_datatype_type_extent(rdtype, &rextent);
        int w_rank = ompi_comm_rank(comm);
        sbuf = ((char*) rbuf) + (ptrdiff_t)w_rank * (ptrdiff_t)rcount * rextent;
        scount = rcount;
        sdtype = rdtype;
    }

    /* Case 1 : proc topology => Block */
    if (han_module->is_mapbycore) {
        low_comm->c_coll->coll_allgather(sbuf, scount, sdtype,
                                        tmp_recv_buf_start, rcount, rdtype,
                                        low_comm, low_comm->c_coll->coll_allgather_module);

        up_comm->c_coll->coll_allgather(tmp_recv_buf_start, rcount*low_size, rdtype,
                                        rbuf, rcount*low_size, rdtype,
                                        up_comm, up_comm->c_coll->coll_allgather_module);
    }
    /* Case 2 : With reordering */
    else {
        low_comm->c_coll->coll_allgather(sbuf, scount, sdtype,
                                        tmp_recv_buf_start, rcount, rdtype,
                                        low_comm, low_comm->c_coll->coll_allgather_module);

        /* Alloc Temporary reorder buffer */
        char *reorder_buf = NULL;
        char *reorder_buf_start = NULL;
        rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * low_size * up_size, &rgap);
        reorder_buf = (char *) malloc(rsize);
        reorder_buf_start = reorder_buf - rgap;

        up_comm->c_coll->coll_allgather(tmp_recv_buf_start, rcount*low_size, rdtype,
                                        reorder_buf_start, rcount*low_size, rdtype,
                                        up_comm, up_comm->c_coll->coll_allgather_module);
        /* Reorder in receive buffer */
        mca_coll_han_reorder_allgather_low(reorder_buf_start,
                                            rbuf, rcount, rdtype,
                                            low_comm, up_comm,
                                            han_module);
        free(reorder_buf);
    }

    free(tmp_recv_buf);

    return OMPI_SUCCESS;
}

/* Reordering Fonction
 * World rank is found via han_module->global_ranks :
 * World rank = global_rank[up_rank * maximum_size[INTRA_NODE]
 *                          + low_rank]
 * Src_shift here is low_comm first => Allgather low algorithm*/
void
mca_coll_han_reorder_allgather_low(const void *sbuf,
                                   void *rbuf, int count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_communicator_t *low_comm,
                                   struct ompi_communicator_t *up_comm,
                                   mca_coll_han_module_t *han_module)
{
    int n_block = 0;
    ptrdiff_t rextent;
    ompi_datatype_type_extent(dtype, &rextent);
    for (int up = 0; up < han_module->maximum_size[INTER_NODE]; up++ ) {
        for (int low=0; low < han_module->maximum_size[LEAF_LEVEL]; low++) {
            ptrdiff_t block_size = rextent * (ptrdiff_t)count;
            ptrdiff_t src_shift = block_size * n_block;
            ptrdiff_t dest_shift = block_size * (ptrdiff_t)han_module->global_ranks[up *
                                                 han_module->maximum_size[LEAF_LEVEL]
                                                 + low];
            ompi_datatype_copy_content_same_ddt(dtype,
                                            (ptrdiff_t)count,
                                            (char *)rbuf + dest_shift,
                                            (char *)sbuf + src_shift);
            n_block++;
        }
    }
}

/* Allgather Up algorithm:
 *
 * Really similar to the previous algorithm except that
 * this time the first allgather is done on up communicators.
 *
 * This inversion allows us to make a smaller allgather
 * on up communicator in order to improve performance.
 *
 * First an allgather is done on all up communicators
 * Then another allgather on all low communicators
 *
 * For a block distribution, there is a reorder but
 * for a cyclic distribution reorder is not needed.
 *
 * Example in cyclic distribution:
 *
 * wrank(up_rank, low_rank)[Send_buf]
 *
 * 0 (0, 0)[A]    1 (1, 0)[B]
 *
 * 2 (0, 1)[C]    3 (1, 1)[D]
 *
 * 1) Allgather on up communicator [Tmp_buf]
 *
 * 0[A, B]        1[A, B]
 *
 * 2[C, D]        3[C, D]
 *
 * 2) Allgather on low communicator  [Recv_buf]
 *
 * 0[A, D, C, B]        1[A, D, C, B]
 *
 * 2[A, D, C, B]        3[A, D, C, B]
 *
 *
 */
int
mca_coll_han_allgather_intra_up(const void *sbuf, int scount,
                                struct ompi_datatype_t *sdtype,
                                void* rbuf, int rcount,
                                struct ompi_datatype_t *rdtype,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module){

    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    /* discovery topology */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). Fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* setup up/low coordinates */
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);

    /* Alloc Temporary recv buffer */
    char *tmp_recv_buf = NULL;
    char *tmp_recv_buf_start = NULL;
    ptrdiff_t rsize, rgap = 0;
    rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * up_size , &rgap);
    tmp_recv_buf = (char *) malloc(rsize);
    tmp_recv_buf_start = tmp_recv_buf - rgap;

    /* MPI IN PLACE */
    if (MPI_IN_PLACE == sbuf) {
        ptrdiff_t rextent;
        ompi_datatype_type_extent(rdtype, &rextent);
        int w_rank = ompi_comm_rank(comm);
        sbuf = ((char*) rbuf) + (ptrdiff_t)w_rank * (ptrdiff_t)rcount * rextent;
        scount = rcount;
        sdtype = rdtype;
    }

    /* Check if there is a cyclic topology */
    mca_coll_han_topo_cyclic(comm, han_module, low_size, up_size);
    if (han_module->is_cyclic) {
        /* Case 1 : proc topology => cyclic */
        up_comm->c_coll->coll_allgather(sbuf, scount, sdtype,
                                        tmp_recv_buf_start, rcount, rdtype,
                                        up_comm, up_comm->c_coll->coll_allgather_module);
        low_comm->c_coll->coll_allgather(tmp_recv_buf_start, rcount*up_size, rdtype,
                                         rbuf, rcount*up_size, rdtype,
                                         low_comm, low_comm->c_coll->coll_allgather_module);
    } else {
        /* Case 2 : With reordering */
        up_comm->c_coll->coll_allgather(sbuf, scount, sdtype,
                                         tmp_recv_buf_start, rcount, rdtype,
                                         up_comm, up_comm->c_coll->coll_allgather_module);
        /* Alloc reorder buffer */
        char *reorder_buf = NULL;
        char *reorder_buf_start = NULL;
        rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * low_size * up_size, &rgap);
        reorder_buf = (char *) malloc(rsize);
        reorder_buf_start = reorder_buf - rgap;
        low_comm->c_coll->coll_allgather(tmp_recv_buf_start, rcount*up_size, rdtype,
                                         reorder_buf_start, rcount*up_size, rdtype,
                                         low_comm, low_comm->c_coll->coll_allgather_module);

        mca_coll_han_reorder_allgather_up(reorder_buf_start,
                                          rbuf, rcount, rdtype,
                                          low_comm, up_comm,
                                          han_module);
        free(reorder_buf);
    }
    free(tmp_recv_buf);
    return OMPI_SUCCESS;
}


/* Reordering Fonction
 * World rank is found via han_module->global_ranks :
 * World rank = global_rank[up_rank * maximum_size[INTRA_NODE]
 *                          + low_rank]
 * Src_shift here is up_comm first => Allgather up algorithm*/
void
mca_coll_han_reorder_allgather_up(const void *sbuf,
                                  void *rbuf, int count,
                                  struct ompi_datatype_t *dtype,
                                  struct ompi_communicator_t *low_comm,
                                  struct ompi_communicator_t *up_comm,
                                  mca_coll_han_module_t *han_module)
{
    int n_block = 0;
    ptrdiff_t rextent;
    ompi_datatype_type_extent(dtype, &rextent);
    for (int low=0; low<han_module->maximum_size[LEAF_LEVEL]; low++) {
        for (int up=0; up<han_module->maximum_size[INTER_NODE]; up++) {
            ptrdiff_t block_size = rextent * (ptrdiff_t)count;
            ptrdiff_t src_shift = block_size * n_block;
            ptrdiff_t dest_shift = block_size * (ptrdiff_t)han_module->global_ranks[up * 
                                                 han_module->maximum_size[LEAF_LEVEL] 
                                                 + low];
            ompi_datatype_copy_content_same_ddt(dtype,
                                            (ptrdiff_t)count,
                                            (char *)rbuf + dest_shift,
                                            (char *)sbuf + src_shift);
            n_block++;
        }
    }
}

/* Allgather Split algorithm:
 *
 * Similar to the simple algorithm except that
 * this time we have multiple roots for the intra node gather so
 * we do more collective but smaller.
 *
 * This algorithm only work in block distribution with an even low_size.
 * We select low_size/2 roots in each low communicators.
 *
 * First we split all low communicators in low_size/2 communicators.
 * All this splitted comms gather data.
 *
 * After that each gather root performs an Allgatherv on its up_comunicators.
 * Here, we use the allgatherv collective displacement to avoid reordering.
 *
 * And then a loop of broadcast on each roots end this allgather
 * We use a vector derivated datatype for these broadcasts to store data right into the rbuf.
 *
 * Currently the number of selected roots is fixed at low_size/2 but later
 * why not select it via a mca parameters.
 *
 * Example:
 *
 * wrank(up_rank, low_rank)[Send_buf]
 *
 * 0(0, 0)[A]    4(1, 0)[E]
 * 1(0, 1)[B]    5(1, 1)[F]
 * 2(0, 2)[C]    6(1, 2)[G]
 * 3(0, 3)[D]    7(1, 3)[H]
 *
 * 1) Gather on each roots selected in low communicators[Tmp_buf]
 *
 * 0[A, B]        4[E, F]
 * 1              5
 * 2[C, D]        6[G, H]
 * 3              7
 *
 * 2) Allgatherv on up communicators[Recv_buf]
 *
 * 0[A, B, _, _, E, F, _, _]        4[A, B, _, _, E, F, _, _]
 * 1[]                              5[]
 * 2[_, _, C, D, _, _, G, H]        6[_, _, C, D, _, _, G, H]
 * 3[]                              7[]
 *
 * 3) Loop of broadcast for each gather roots(Low_size/2)[Recv_buf]
 *
 * First broadcast on each low_rank == 0
 * 
 * 0[A, B, _, _, E, F, _, _]        4[A, B, _, _, E, F, _, _]
 * 1[A, B, _, _, E, F, _, _]        5[A, B, _, _, E, F, _, _]
 * 2[A, B, C, D, E, F, G, H]        6[A, B, C, D, E, F, G, H]
 * 3[A, B, _, _, E, F, _, _]        7[A, B, _, _, E, F, _, _]
 *
 * Then last broadcast with second root( low_rank == 2)
 *
 * 0[A, B, C, D, E, F, G, H]        4[A, B, C, D, E, F, G, H]
 * 1[A, B, C, D, E, F, G, H]        5[A, B, C, D, E, F, G, H]
 * 2[A, B, C, D, E, F, G, H]        6[A, B, C, D, E, F, G, H]
 * 3[A, B, C, D, E, F, G, H]        7[A, B, C, D, E, F, G, H]
 *
 */
int
mca_coll_han_allgather_intra_split(const void *sbuf, int scount,
                                   struct ompi_datatype_t *sdtype,
                                   void* rbuf, int rcount,
                                   struct ompi_datatype_t *rdtype,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module)
{
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }

    /* discovery topology */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). Fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }
    /* Algorithm only work in block disribution */
    if (!han_module->is_mapbycore) {
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* setup up/low coordinates */
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);
    /* Fixed size for now */
    int split_size = 2; // size of each split communicator
    /* Check that the size is divisible by the gather_count */
    if (low_size % split_size) {
        /* Select simple algorithm instead */
        han_module->have_even_low_size = false;
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype,
                                            comm, comm->c_coll->coll_allgather_module);
    }
    int i;
    /* allocate the intermediary buffer
     * to gather on leaders on the low sub communicator*/
    ptrdiff_t rext;
    ompi_datatype_type_extent(rdtype, &rext);
    char *tmp_buf = NULL;
    char *tmp_buf_start = NULL;
    ptrdiff_t rsize, rgap = 0;
    ompi_communicator_t *low_split = han_module->cached_allgather_split_comms;
    MPI_Datatype broadcast_vector;

    /* Compute the size to receive all the local data, including datatypes empty gaps */
    rsize = opal_datatype_span(&rdtype->super, (int64_t)rcount * split_size, &rgap);

    /* intermediary buffer on node roots to gather on low comm */
    tmp_buf = (char *) malloc(rsize);
    tmp_buf_start = tmp_buf - rgap;

    /* MPI IN PLACE */
    if (MPI_IN_PLACE == sbuf) {
        int w_rank = ompi_comm_rank(comm);
        sbuf = ((char*) rbuf) + (ptrdiff_t)w_rank * (ptrdiff_t)rcount * rext;
        scount = rcount;
        sdtype = rdtype;
    }

    /* 1. low gather on low_comm roots into tmp_buf */
    if (NULL == low_split) {
        int color;
        color = low_rank/split_size;
        ompi_comm_split(low_comm, color, low_rank, &low_split, false);
    }
    low_split->c_coll->coll_gather((char *)sbuf, scount, sdtype,
                                   tmp_buf_start, rcount, rdtype, 0,
                                   low_split, low_split->c_coll->coll_gather_module);

    /* 2. allgather between node leaders, from tmp_buf to rbuf */
    if (low_rank%split_size == 0) {
        int *displ, *allgatherv_count;
        int up;
        /* 2a. inter node allgatherv on each roots*/
        displ = malloc(up_size * sizeof(int));
        allgatherv_count = malloc(up_size * sizeof(int));
        for (up=0; up<up_size; up++) {
            /* displ => rcount * split_size (size of one block after gather) *
             * low_size / split_size (all gather blocks in low_comm) * up  */
            displ[up] = rcount * low_size * up;
            allgatherv_count[up] = rcount * split_size;
        }
        up_comm->c_coll->coll_allgatherv(tmp_buf_start, rcount*split_size, rdtype,
                                         (char *)rbuf+low_rank*rext*rcount, allgatherv_count,
                                         displ, rdtype,
                                         up_comm, up_comm->c_coll->coll_allgatherv_module);
        free(displ);
        free(allgatherv_count);
    }

    free(tmp_buf);
    tmp_buf = NULL;
    tmp_buf_start = NULL;

    /* 3. up broadcast: leaders broadcast on their nodes */
    ompi_datatype_create_vector(up_size, rcount*split_size, rcount*low_size, rdtype, &broadcast_vector);
    ompi_datatype_commit(&broadcast_vector);
    if(mca_coll_han_component.allgather_split_ibcast) {
        struct ompi_request_t **ibcast_rq;
        ibcast_rq = malloc((low_size/split_size) * sizeof(struct ompi_request_t *));
        for (i=0; i<low_size/split_size; i++) {
            low_comm->c_coll->coll_ibcast((char *)rbuf + i*rext*split_size*rcount, 1, broadcast_vector,
                                          split_size*i, low_comm, ibcast_rq + i,
                                          low_comm->c_coll->coll_bcast_module);
        }
        /* Wait for completion */
        ompi_request_wait_all(low_size/split_size, ibcast_rq, MPI_STATUSES_IGNORE);
        free(ibcast_rq);
    } else {
        for (i=0; i<low_size/split_size; i++) {
            low_comm->c_coll->coll_bcast((char *)rbuf + i*rext*split_size*rcount, 1, broadcast_vector,
                                         split_size*i, low_comm,
                                         low_comm->c_coll->coll_bcast_module);
        }
    }
    ompi_datatype_destroy(&broadcast_vector);
    return OMPI_SUCCESS;
}

/**
 * Implementation of the recursive allgather by calling gather, allgather and broadcast modules
 *
 * For a block distribution, no reorder is needed.
 *
 * Example without reordering (Block distribution): 2 clusters, 2 nodes per cluster, 2 ranks MPI per
 * node
 *
 * wrank[Send_buf][Recv_buf]
 *
 * cluster 1                                            cluster 2
 * 0[A][]       2[C][]                                  4[E][]              6[G][]
 *
 * 1[B][]       3[D][]                                  5[F][]              7[H][]
 *
 * 1) gather in leaf level
 * 0 is the global root
 *
 * cluster 1                                            cluster 2
 * 0[A][A,B]        2[C][C,D]                           4[E][E,F]           6[G][G,H]
 *
 * 1[B][]           3[D][]                              5[F][]              7[H][]
 *
 * 2) gather in inter node
 * New sub roots do a gather
 *
 * cluster 1                                            cluster 2
 * 0[A,B][A,B,C,D]      2[C,D][]                        4[E,F][E,F,G,H]     6[G,H][]
 *
 * 1[][]                3[][]                           5[][]               7[][]
 *
 * 3) allgather in GATEWAY
 *
 * cluster 1                                            cluster 2
 * 0[A,B,C,D][A,B,C,D,E,F,G,H]      2[][]               4[E,F,G,H][A,B,C,D,E,F,G,H]     6[][]
 *
 * 1[][]                            3[][]               5[][]                           7[][]
 *
 * 4) broadcast in inter node
 * Wrank[buf]
 * In cluster 1, process 0 sends its data to process 2
 * In cluster 2, process 4 sends its data to process 6
 *
 * cluster 1                                            cluster 2
 * 0[A,B,C,D,E,F,G,H]     2[recvbuf]                    4[A,B,C,D,E,F,G,H]    6[recvbuf]
 *
 * 1[]                    3[]                           5[]                   7[]
 *
 * 5) broadcast in leaf level
 * wrank[buf]
 * In cluster 1, process 0 sends its data to process 1, and process 2 sends its data to process 3
 * In cluster 2, process 4 sends its data to process 5, and process 6 sends its data to process 7
 *
 * cluster 1                                            cluster 2
 * 0[A,B,C,D,E,F,G,H]       2[A,B,C,D,E,F,G,H]          4[A,B,C,D,E,F,G,H]      6[A,B,C,D,E,F,G,H]
 *
 * 1[recvbuf]               3[recvbuf]                  5[recvbuf]              7[recvbuf]
 */
int mca_coll_han_allgather_recursive_gather_bcast(const void *sbuf, int scount,
                                                  struct ompi_datatype_t *sdtype, void *rbuf,
                                                  int rcount, struct ompi_datatype_t *rdtype,
                                                  struct ompi_communicator_t *comm,
                                                  mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;

    /* create the subcommunicators */
    if (OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on "
                             "another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm,
                                            comm->c_coll->coll_allgather_module);
    }

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm,
                                            comm->c_coll->coll_allgather_module);
    }

    /* Global communicator information */
    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);

    /* Aliases */
    ompi_communicator_t **sub_comm = han_module->sub_comm;
    const int *my_sub_rank = mca_coll_han_get_sub_ranks(han_module, w_rank);

    /* Search top levels */
    int allgather_lvl;
    for (allgather_lvl = GLOBAL_COMMUNICATOR-1; allgather_lvl > LEAF_LEVEL; allgather_lvl--) {
        if (1 < ompi_comm_size(sub_comm[allgather_lvl])) {
            break;
        }
    }

    int top_comm_lvl = LEAF_LEVEL;
    int top_nb_blocks = 1;
    if (0 == my_sub_rank[top_comm_lvl]) {
        for (top_comm_lvl = LEAF_LEVEL+1;
             top_comm_lvl < GLOBAL_COMMUNICATOR;
             top_comm_lvl++) {
            top_nb_blocks *= ompi_comm_size(sub_comm[top_comm_lvl-1]);
            if (0 != my_sub_rank[top_comm_lvl]) {
                break;
            }
        }
    }
    /* Rank 0 workaround */
    if (GLOBAL_COMMUNICATOR == top_comm_lvl) {
        do {
            top_comm_lvl--;
        } while (1 == ompi_comm_size(sub_comm[top_comm_lvl]));
        top_nb_blocks = w_size/ompi_comm_size(sub_comm[top_comm_lvl]);
    }

    /* Reorder buffer management
     * Directly store in rbuf when no reorder is needed
     */
    char *tmp_buf = NULL;
    char *tmp_buf_start = NULL;
    int start_offset;
    if (han_module->is_mapbycore) {
        tmp_buf_start = (char *) rbuf;
    } else if (LEAF_LEVEL < top_comm_lvl) {
        ptrdiff_t rsize = 0;
        ptrdiff_t rgap = 0;

        if (allgather_lvl <= top_comm_lvl) {
            rsize = opal_datatype_span(&rdtype->super,
                                       (size_t) rcount * w_size,
                                       &rgap);
        } else {
            rsize = opal_datatype_span(&rdtype->super,
                                       (size_t) rcount * top_nb_blocks,
                                       &rgap);
        }
        tmp_buf = (char *) malloc(rsize);
        tmp_buf_start = tmp_buf - rgap;
    }

    if (allgather_lvl <= top_comm_lvl) {
        start_offset = my_sub_rank[allgather_lvl] * top_nb_blocks;
    } else {
        start_offset = 0;
    }

    ptrdiff_t extent;
    ompi_datatype_type_extent(rdtype, &extent);

    if (MPI_IN_PLACE == sbuf && !han_module->is_mapbycore) {
        scount = rcount;
        sdtype = rdtype;
        sbuf = ((char *) rbuf) + (ptrdiff_t)(w_rank) * (ptrdiff_t) scount * extent;
    }

    /* LEAF_LEVEL gather */
    sub_comm[LEAF_LEVEL]->c_coll->coll_gather(sbuf, scount, sdtype,
                                              tmp_buf_start + start_offset * rcount * extent,
                                              rcount, rdtype, 0, sub_comm[LEAF_LEVEL],
                                              sub_comm[LEAF_LEVEL]->c_coll->coll_gather_module);

    /* Track if rank will be in top allgather */
    bool rank_ispresent = false;

    /* Main gather loop */
    int nb_blocks = ompi_comm_size(sub_comm[LEAF_LEVEL]);
    int topo_lvl = LEAF_LEVEL;
    if (0 == ompi_comm_rank(sub_comm[LEAF_LEVEL])) {
        for (topo_lvl = LEAF_LEVEL + 1; topo_lvl < allgather_lvl; topo_lvl++) {

            /* Skip self levels */
            if(1 == ompi_comm_size(sub_comm[topo_lvl])){
                rank_ispresent = true;
                continue;
            }

            /* Root handles data for upper levels */
            if (0 == my_sub_rank[topo_lvl]) {
                sub_comm[topo_lvl]
                    ->c_coll->coll_gather(MPI_IN_PLACE, 0, NULL,
                                          tmp_buf_start + start_offset * rcount * extent,
                                          nb_blocks * rcount, rdtype,
                                          0, sub_comm[topo_lvl],
                                          sub_comm[topo_lvl]->c_coll->coll_gather_module);
                rank_ispresent = true;
            } else {
                sub_comm[topo_lvl]
                    ->c_coll->coll_gather(tmp_buf_start + start_offset * rcount * extent,
                                          nb_blocks * rcount, rdtype,
                                          NULL, 0, NULL,
                                          0, sub_comm[topo_lvl],
                                          sub_comm[topo_lvl]->c_coll->coll_gather_module);
                rank_ispresent = false;
                break;
            }

            nb_blocks *= ompi_comm_size(sub_comm[topo_lvl]);
        }
    }

    /* Top level allgather */
    if (rank_ispresent) { 
        sub_comm[allgather_lvl]
            ->c_coll->coll_allgather(MPI_IN_PLACE, 0, NULL, tmp_buf_start,
                                     top_nb_blocks * rcount, rdtype,
                                     sub_comm[allgather_lvl],
                                     sub_comm[allgather_lvl]->c_coll->coll_allgather_module);

        /* Reorder data on roots before broadcasting */
        if (!han_module->is_mapbycore) {
            ptrdiff_t rextent;
            ompi_datatype_type_extent(rdtype, &rextent);
            for (int topo_rank = 0; topo_rank < w_size; topo_rank++) {
                ptrdiff_t block_size = rextent * (ptrdiff_t)rcount;
                ptrdiff_t src_shift = block_size * topo_rank;
                ptrdiff_t dest_shift = block_size * han_module->global_ranks[topo_rank];
                ompi_datatype_copy_content_same_ddt(rdtype, (ptrdiff_t) rcount,
                                                    (char *) rbuf + dest_shift,
                                                    tmp_buf_start + src_shift);
            }
        }

        topo_lvl = allgather_lvl-1;
    }

    /* Allgather done on top level, start diving */
    for (; topo_lvl >= LEAF_LEVEL; topo_lvl--) {
        sub_comm[topo_lvl]
            ->c_coll->coll_bcast(rbuf, rcount * w_size, rdtype, 0,
                                 sub_comm[topo_lvl],
                                 sub_comm[topo_lvl]->c_coll->coll_bcast_module);
    }

    free(tmp_buf);

    return OMPI_SUCCESS;
}


/**
 * Descending Allgather algorithm:
 *
 * A first allgatherv is done on the top level using both sbuf and rbuf.
 * Then, an allgatherv is called in-place on the rbuf, and we climb down the
 * topology tree calling an allgatherv on each level.
 * MPI_COMM_SELF levels are skiped.
 * Vector datatypes are used to avoid reordering.
 *
 * Example: 2 nodes, 2 sockets per node, 2 ranks per socket, distributed as follow
 *
 * Node 0                 Node 1                    
 * Socket 0   Socket 1    Socket 0   Socket 1
 *  Rank 0     Rank 2      Rank 4     Rank 6
 *  Rank 1     Rank 3      Rank 5     Rank 7
 *
 *
 * At the begginning of the collective, each rank has:
 *  - its contribution in its sbuf
 *  - an empty rbuf
 *
 * 0[A][x,x,x,x,x,x,x,x]   2[C][x,x,x,x,x,x,x,x]    4[E][x,x,x,x,x,x,x,x]   6[G][x,x,x,x,x,x,x,x]
 * 1[B][x,x,x,x,x,x,x,x]   3[D][x,x,x,x,x,x,x,x]    5[F][x,x,x,x,x,x,x,x]   7[H][x,x,x,x,x,x,x,x]
 *
 * Here, the gateway level is an MPI_COMM_SELF level, skip it.
 *
 * On the INTER_NODE level, ranks share their data with their peer
 * through an allgatherv. Allgatherv is used instead of allgather even
 * if data has the same size: the displacement allows the buffer to be
 * received directly in its final position in the rbuf.
 * As it is the first collective, simply use sbuf as send buffer.
 *
 *   _____________________________________________
 *  /                                             \
 * 0[A][A,x,x,x,E,x,x,x]  2[C][x,x,C,x,x,x,G,x]    4[E][A,x,x,x,E,x,x,x]  6[G][x,x,C,x,x,x,G,x]
 *                         \_____________________________________________/
 *   _____________________________________________
 *  /                                             \
 * 1[B][x,B,x,x,x,F,x,x]  3[D][x,x,x,D,x,x,x,H]    5[F][x,B,x,x,x,F,x,x]  7[H][x,x,x,D,x,x,x,H]
 *                          \_____________________________________________/
 *
 * Following iterations are similar to each other: an in-place allgaterv
 * is performed on each non-self topological level, directly in rbuf.
 *
 * INTRA_NODE:
 * 0[A,x,C,x,E,x,G,x]<->2[A,x,C,x,E,x,G,x]    4[A,x,C,x,E,x,G,x]<->6[A,x,C,x,E,x,G,x]
 * 1[x,B,x,D,x,F,x,H]<->3[x,B,x,D,x,F,x,H]    5[x,B,x,D,x,F,x,H]<->7[x,B,x,D,x,F,x,H]
 *
 * LEAF_LEVEL:
 * 0[A,B,C,D,E,F,G,H]   2[A,B,C,D,E,F,G,H]    4[A,B,C,D,E,F,G,H]   6[A,B,C,D,E,F,G,H]
 * |                    |                     |                    |
 * 1[A,B,C,D,E,F,G,H]   3[A,B,C,D,E,F,G,H]    5[A,B,C,D,E,F,G,H]   7[A,B,C,D,E,F,G,H]
 *
 * To send data fragmented in rbuf, one vector datatype
 * is built for each topological level.
 * Those vectors are resized in order to be intertwined.
 *
 * Limitations:
 *  - This algorithm can only handle block distribution (is_mapbycore)
 *  - This algorithm requires a perfectly balanced topology on every levels.
 * Otherwise, it falls back on another algorithm or component.
 */ 
int mca_coll_han_allgather_recursive_descending(const void *sbuf, int scount,
                                                struct ompi_datatype_t *sdtype, void *rbuf,
                                                int rcount, struct ompi_datatype_t *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module)
{
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module; 

    if (OMPI_SUCCESS != mca_coll_han_comm_create_multi_level(comm, han_module)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator. Fall back on "
                             "another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm,
                                            comm->c_coll->coll_allgather_module);
    }

    /* unbalanced case needs algorithm adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allgather);
        return comm->c_coll->coll_allgather(sbuf, scount, sdtype, rbuf, rcount, rdtype, comm,
                                            comm->c_coll->coll_allgather_module);
    }

    if (!han_module->is_mapbycore) {
        return mca_coll_han_allgather_recursive_gather_bcast(sbuf, scount, sdtype,
                                                             rbuf, rcount, rdtype,
                                                             comm, module);
    }

    /* GLOBAL_COMMUNICATOR info */
    int w_rank = ompi_comm_rank(comm);

    /* Aliases */
    ompi_communicator_t **sub_comm = han_module->sub_comm;
    const int *sub_ranks = mca_coll_han_get_sub_ranks(han_module, w_rank);

    int top_comm_lvl;
    for (top_comm_lvl = GLOBAL_COMMUNICATOR-1;
         LEAF_LEVEL < top_comm_lvl && 1 >= han_module->maximum_size[top_comm_lvl];
         top_comm_lvl--); /* Empty loop */

    /* Store sub_ranks of remote lowest owned buffer to compute rdispls */
    /* For each next sub-communicators, each process broadcasts all the data
     * it knows in a single vector. It contains N blocks which is the cumulated
     * product of each descended communicators sizes. The new vector stride
     * is divided each time by last the subcomm size. This subranks array will
     * help to compute the offset in the receive buffer of the first vector block.
     */
    int first_known_block_sub_ranks[NB_TOPO_LVL-1];
    for (int topo_lvl = LEAF_LEVEL; topo_lvl < GLOBAL_COMMUNICATOR; topo_lvl++) {
        first_known_block_sub_ranks[topo_lvl] = sub_ranks[topo_lvl];
    }

    /* Maximum sub_comm size to allocate rcounts and rdispls */
    int max_sub_comm_size = 1;
    for (int topo_lvl = LEAF_LEVEL; topo_lvl < GLOBAL_COMMUNICATOR; topo_lvl++) {
        int sub_comm_size = han_module->maximum_size[topo_lvl];
        if (sub_comm_size > max_sub_comm_size) {
            max_sub_comm_size = sub_comm_size;
        }
    }

    int *rcounts = malloc(max_sub_comm_size * sizeof(int));
    int *rdispls = malloc(max_sub_comm_size * sizeof(int));

    /* Only one block per peer on top level */
    for (int sub_rank = 0; sub_rank < max_sub_comm_size; sub_rank++) {
        rcounts[sub_rank] = rcount;
    }

    /* Place data in the right place to perform in_place gathering on lower levels */
    for (int sub_rank = 0; sub_rank < han_module->maximum_size[top_comm_lvl]; sub_rank++) {
        first_known_block_sub_ranks[top_comm_lvl] = sub_rank;
        int remote_wrank = mca_coll_han_get_global_rank(han_module, first_known_block_sub_ranks);
        rdispls[sub_rank] = remote_wrank * rcount;
    }

    /* Perform allgatherv on top level */
    sub_comm[top_comm_lvl]->c_coll->coll_allgatherv(sbuf, scount, sdtype,
                                                    rbuf, rcounts, rdispls, rdtype,
                                                    sub_comm[top_comm_lvl],
                                                    sub_comm[top_comm_lvl]->c_coll->coll_allgatherv_module);

    /* We gathered over this level */
    first_known_block_sub_ranks[top_comm_lvl] = 0;

    /* Blocks management is done through vectors on lower levels */
    for (int sub_rank = 0; sub_rank < max_sub_comm_size; sub_rank++) {
        rcounts[sub_rank] = 1;
    }

    /* Magic here */
    int n_blocks = han_module->maximum_size[top_comm_lvl];
    int stride = ompi_comm_size(comm)/n_blocks;

    /* Get rtype extent to ease vector management */
    ptrdiff_t rextent;
    ompi_datatype_type_extent(rdtype, &rextent);

    /* Allgatherv loop on lower levels */
    for (int topo_lvl = top_comm_lvl-1; topo_lvl >= LEAF_LEVEL; topo_lvl--) {
        int sub_comm_size = han_module->maximum_size[topo_lvl];

        /* Skip self levels */
        if (1 == sub_comm_size) {
            continue;
        }

        /* Compute displacements */
        for (int sub_rank = 0; sub_rank < sub_comm_size; sub_rank++) {
            first_known_block_sub_ranks[topo_lvl] = sub_rank;
            rdispls[sub_rank] = mca_coll_han_get_global_rank(han_module, first_known_block_sub_ranks);
        }

        /* Build a vector and resize it */
        struct ompi_datatype_t *vector_not_resized;
        struct ompi_datatype_t *vector_resized;

        /* Create vector to represent data */
        ompi_datatype_create_vector(n_blocks, rcount, stride*rcount, rdtype, &vector_not_resized);
        ompi_datatype_commit(&vector_not_resized);

        /* Resize vector to intertwine data */
        ompi_datatype_create_resized(vector_not_resized, 0, rcount * rextent, &vector_resized);
        ompi_datatype_commit(&vector_resized);

        /* Call allgatherv on that level */
        sub_comm[topo_lvl]->c_coll->coll_allgatherv(MPI_IN_PLACE, 0, NULL,
                                                    rbuf, rcounts, rdispls,
                                                    vector_resized,
                                                    sub_comm[topo_lvl],
                                                    sub_comm[topo_lvl]->c_coll->coll_allgatherv_module);

        /* Release datatypes */
        ompi_datatype_destroy(&vector_not_resized);
        ompi_datatype_destroy(&vector_resized);

        /* We gathered over this level */
        first_known_block_sub_ranks[topo_lvl] = 0;

        /* More blocks, lower stride */
        stride /= sub_comm_size;
        n_blocks *= sub_comm_size;
    }

    free(rcounts);
    free(rdispls);

    return OMPI_SUCCESS;
}
