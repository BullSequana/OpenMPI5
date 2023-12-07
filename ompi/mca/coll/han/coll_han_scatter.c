/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2022      IBM Corporation. All rights reserved
 * Copyright (c) 2020-2023 BULL S.A.S. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

#define CHECK(val) do {               \
    if (val != OMPI_SUCCESS) {        \
        line = __LINE__;              \
        goto end;                     \
    }                                 \
} while (0)

#define MALLOC(ptr, size) do {                  \
    ptr = malloc(size);                         \
    if (NULL == ptr) {                          \
        err = OMPI_ERR_OUT_OF_RESOURCE;         \
        goto end;                               \
    }                                           \
} while (0)

/*
 * @file
 *
 * This files contains all the hierarchical implementations of scatter
 */

static int mca_coll_han_scatter_us_task(void *task_args);
static int mca_coll_han_scatter_ls_task(void *task_args);

static inline void
ompi_coll_han_reorder_scatter(const void *sbuf,
                   void *rbuf, int count,
                   struct ompi_datatype_t *dtype,
                   struct ompi_communicator_t *comm,
                   int * topo);
/* Only work with regular situation (each node has equal number of processes) */

static inline void
mca_coll_han_set_scatter_args(mca_coll_han_scatter_args_t * args,
                              mca_coll_task_t * cur_task,
                              void *sbuf,
                              void *sbuf_inter_free,
                              void *sbuf_reorder_free,
                              int scount,
                              struct ompi_datatype_t *sdtype,
                              void *rbuf,
                              int rcount,
                              struct ompi_datatype_t *rdtype,
                              int root,
                              int root_up_rank,
                              int root_low_rank,
                              struct ompi_communicator_t *up_comm,
                              struct ompi_communicator_t *low_comm,
                              int w_rank, bool noop, ompi_request_t * req)
{
    args->cur_task = cur_task;
    args->sbuf = sbuf;
    args->sbuf_inter_free = sbuf_inter_free;
    args->sbuf_reorder_free = sbuf_reorder_free;
    args->scount = scount;
    args->sdtype = sdtype;
    args->rbuf = rbuf;
    args->rcount = rcount;
    args->rdtype = rdtype;
    args->root = root;
    args->root_up_rank = root_up_rank;
    args->root_low_rank = root_low_rank;
    args->up_comm = up_comm;
    args->low_comm = low_comm;
    args->w_rank = w_rank;
    args->noop = noop;
    args->req = req;
}

/*
 * Main function for taskified scatter:
 * after data reordering, calls us task, a scatter on up communicator
 */
int
mca_coll_han_scatter_intra(const void *sbuf, int scount,
                           struct ompi_datatype_t *sdtype,
                           void *rbuf, int rcount,
                           struct ompi_datatype_t *rdtype,
                           int root,
                           struct ompi_communicator_t *comm, mca_coll_base_module_t * module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;
    int w_rank, w_size;
    w_rank = ompi_comm_rank(comm);
    w_size = ompi_comm_size(comm);

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                          comm, comm->c_coll->coll_scatter_module);
    }

    /* Create the subcommunicators */
    if( OMPI_SUCCESS != mca_coll_han_comm_create(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator. Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                            comm, han_module->previous_scatter_module);
    }

    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    int* topo = mca_coll_han_topo_init(comm, han_module, 2);
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator (imbalance). Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return han_module->previous_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                            comm, han_module->previous_scatter_module);
    }

    ompi_communicator_t *low_comm =
        han_module->cached_low_comms[mca_coll_han_component.han_scatter_low_module];
    ompi_communicator_t *up_comm =
        han_module->cached_up_comms[mca_coll_han_component.han_scatter_up_module];
    int *vranks = han_module->cached_vranks;
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);

    /* Set up request */
    ompi_request_t *temp_request = OBJ_NEW(ompi_request_t);
    temp_request->req_state = OMPI_REQUEST_ACTIVE;
    temp_request->req_type = OMPI_REQUEST_COLL;
    temp_request->req_free = ompi_coll_han_request_free;
    temp_request->req_status = (ompi_status_public_t){0};
    temp_request->req_complete = REQUEST_PENDING;

    int root_low_rank;
    int root_up_rank;

    mca_coll_han_get_ranks(vranks, root, low_size, &root_low_rank, &root_up_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d]: Han Scatter root %d root_low_rank %d root_up_rank %d\n", w_rank,
                         root, root_low_rank, root_up_rank));

    /* Reorder sbuf based on rank.
     * Suppose, message is 0 1 2 3 4 5 6 7
     * and the processes are mapped on 2 nodes (the processes on the node 0 is 0 2 4 6 and the processes on the node 1 is 1 3 5 7),
     * so the message needs to be reordered to 0 2 4 6 1 3 5 7
     */
    char *reorder_buf = NULL;
    char *reorder_sbuf = NULL;

    if (w_rank == root) {
        /* If the processes are mapped-by core, no need to reorder */
        if (han_module->is_mapbycore) {
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                 "[%d]: Han Scatter is_bycore: ", w_rank));
            reorder_sbuf = (char *) sbuf;
        } else {
            ptrdiff_t ssize, sgap = 0, sextent;
            ompi_datatype_type_extent(sdtype, &sextent);
            ssize = opal_datatype_span(&sdtype->super, (int64_t) scount * w_size, &sgap);
            reorder_buf = (char *) malloc(ssize);
            reorder_sbuf = reorder_buf - sgap;
            for (int i = 0; i < up_size; i++) {
                for (int j = 0; j < low_size; j++) {
                    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                         "[%d]: Han Scatter copy from %d %d\n", w_rank,
                                         (i * low_size + j) * 2 + 1,
                                         topo[(i * low_size + j) * 2 + 1]));
                    ompi_datatype_copy_content_same_ddt(sdtype, (ptrdiff_t) scount,
                                                        reorder_sbuf + sextent * (i * low_size +
                                                                                  j) *
                                                        (ptrdiff_t) scount,
                                                        (char *) sbuf +
                                                        sextent *
                                                        (ptrdiff_t) topo[(i * low_size + j) * 2 +
                                                                         1] * (ptrdiff_t) scount);
                }
            }
        }
    }

    /* Create us task */
    mca_coll_task_t *us = OBJ_NEW(mca_coll_task_t);
    /* Setup us task arguments */
    mca_coll_han_scatter_args_t *us_args = malloc(sizeof(mca_coll_han_scatter_args_t));
    mca_coll_han_set_scatter_args(us_args, us, reorder_sbuf, NULL, reorder_buf, scount, sdtype,
                                  (char *) rbuf, rcount, rdtype, root, root_up_rank, root_low_rank,
                                  up_comm, low_comm, w_rank, low_rank != root_low_rank,
                                  temp_request);
    /* Init us task */
    init_task(us, mca_coll_han_scatter_us_task, (void *) (us_args));
    /* Issure us task */
    issue_task(us);

    ompi_request_wait(&temp_request, MPI_STATUS_IGNORE);
    return OMPI_SUCCESS;

}

/* us: upper level (intra-node) scatter task */
int mca_coll_han_scatter_us_task(void *task_args)
{
    mca_coll_han_scatter_args_t *t = (mca_coll_han_scatter_args_t *) task_args;

    if (t->noop) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] Han Scatter:  us noop\n",
                             t->w_rank));
    } else {
        size_t count;
        ompi_datatype_t *dtype;
        if (t->w_rank == t->root) {
            dtype = t->sdtype;
            count = t->scount;
        } else {
            dtype = t->rdtype;
            count = t->rcount;
        }
        int low_size = ompi_comm_size(t->low_comm);
        ptrdiff_t rsize, rgap = 0;
        rsize = opal_datatype_span(&dtype->super, (int64_t) count * low_size, &rgap);
        char *tmp_buf = (char *) malloc(rsize);
        char *tmp_rbuf = tmp_buf - rgap;
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "[%d] Han Scatter:  us scatter\n", t->w_rank));
        /* Inter node scatter */
        size_t rcount;
        ompi_datatype_type_size(dtype, &rcount);
        rcount *= count * low_size;
        t->up_comm->c_coll->coll_scatter((char*) t->sbuf, t->scount * low_size, t->sdtype,
                                         tmp_rbuf, (int) rcount, MPI_BYTE,
                                         t->root_up_rank, t->up_comm,
                                         t->up_comm->c_coll->coll_scatter_module);

        t->sbuf = tmp_rbuf;
        t->sbuf_inter_free = tmp_buf;
    }

    if (t->sbuf_reorder_free != NULL && t->root == t->w_rank) {
        free(t->sbuf_reorder_free);
        t->sbuf_reorder_free = NULL;
    }
    /* Create ls tasks for the current union segment */
    mca_coll_task_t *ls = t->cur_task;
    /* Init ls task */
    init_task(ls, mca_coll_han_scatter_ls_task, (void *) t);
    /* Issue ls task */
    issue_task(ls);

    return OMPI_SUCCESS;
}

/* ls: lower level (shared memory or intra-node) scatter task */
int mca_coll_han_scatter_ls_task(void *task_args)
{
    mca_coll_han_scatter_args_t *t = (mca_coll_han_scatter_args_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] Han Scatter:  ls\n",
                         t->w_rank));
    OBJ_RELEASE(t->cur_task);
    size_t count;
    const struct ompi_datatype_t *dtype;
    if (t->w_rank == t->root) {
        dtype = t->sdtype;
        count = t->scount;
    } else {
        dtype = t->rdtype;
        count = t->rcount;
    }

    size_t dtype_size;
    ompi_datatype_type_size(dtype, &dtype_size);
    t->low_comm->c_coll->coll_scatter((char *) t->sbuf, (int) (count * dtype_size), MPI_BYTE,
                                      (char *) t->rbuf, t->rcount, t->rdtype,
                                      t->root_low_rank, t->low_comm,
                                      t->low_comm->c_coll->coll_scatter_module);

    if (t->sbuf_inter_free != NULL && t->noop != true) {
        free(t->sbuf_inter_free);
        t->sbuf_inter_free = NULL;
    }
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d] Han Scatter:  ls finish\n",
                         t->w_rank));
    ompi_request_t *temp_req = t->req;
    free(t);
    ompi_request_complete(temp_req, 1);
    return OMPI_SUCCESS;
}


int
mca_coll_han_scatter_intra_simple(const void *sbuf, int scount,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  int root,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t * module)
{
    int w_rank, w_size;
    struct ompi_datatype_t * dtype;
    int count;

    w_rank = ompi_comm_rank(comm);
    w_size = ompi_comm_size(comm);

    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                          comm, comm->c_coll->coll_scatter_module);
    }

    /* create the subcommunicators */
    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allgather within this communicator."
                             " Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                            comm, han_module->previous_scatter_module);
    }
    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    int *topo = mca_coll_han_topo_init(comm, han_module, 2);
    if (han_module->are_ppn_imbalanced){
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator. It needs to fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                            comm, han_module->previous_scatter_module);
    }
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* Get the 'virtual ranks' mapping corresponding to the communicators */
    int *vranks = han_module->cached_vranks;
    /* information about sub-communicators */
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    /* Get root ranks for low and up comms */
    int root_low_rank, root_up_rank; /* root ranks for both sub-communicators */
    mca_coll_han_get_ranks(vranks, root, low_size, &root_low_rank, &root_up_rank);

    if (w_rank == root) {
        dtype = sdtype;
        count = scount;
    } else {
        dtype = rdtype;
        count = rcount;
    }

    /* allocate buffer to store unordered result on root
     * if the processes are mapped-by core, no need to reorder:
     * distribution of ranks on core first and node next,
     * in a increasing order for both patterns */
    char *reorder_buf = NULL;  /* allocated memory */
    size_t block_size;

    ompi_datatype_type_size(dtype, &block_size);
    block_size *= count;

    if (w_rank == root) {
        int is_contiguous = ompi_datatype_is_contiguous_memory_layout(dtype, count);

        if (han_module->is_mapbycore && is_contiguous) {
            /* The copy of the data is avoided */
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                 "[%d]: Han scatter: no need to reorder: ", w_rank));
            reorder_buf = (char *)sbuf;
        } else {
            /* Data must be copied, let's be efficient packing it */
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                 "[%d]: Han scatter: needs reordering or compacting: ", w_rank));

            reorder_buf = malloc(block_size * w_size);
            if ( NULL == reorder_buf){
                return OMPI_ERROR;
            }

            /** Reorder and packing:
             * Suppose, the message is 0 1 2 3 4 5 6 7 but the processes are
             * mapped on 2 nodes, for example |0 2 4 6| |1 3 5 7|. The messages to
             * leaders must be 0 2 4 6 and 1 3 5 7.
             * So the upper scatter must send 0 2 4 6 1 3 5 7.
             * In general, the topo[i*topolevel +1]  must be taken.
             */
            ptrdiff_t extent, block_extent;
            ompi_datatype_type_extent(dtype, &extent);
            block_extent = extent * (ptrdiff_t)count;

            for(int i = 0 ; i < w_size ; ++i){
                ompi_datatype_sndrcv((char*)sbuf + block_extent*topo[2*i+1], count, dtype,
                                     reorder_buf + block_size*i, block_size, MPI_BYTE);
            }
            dtype = MPI_BYTE;
            count = block_size;
        }
    }

    /* allocate the intermediary buffer
     * to scatter from leaders on the low sub communicators */

    char *tmp_buf = NULL; /* allocated memory */
    if (low_rank == root_low_rank) {
        tmp_buf = (char *) malloc(block_size * low_size);

        /* 1. up scatter (internode) between node leaders */
        up_comm->c_coll->coll_scatter((char*) reorder_buf,
                    count * low_size,
                    dtype,
                    (char *)tmp_buf,
                    block_size * low_size,
                    MPI_BYTE,
                    root_up_rank,
                    up_comm,
                    up_comm->c_coll->coll_scatter_module);
    }

    /* 2. low scatter on nodes leaders */
    low_comm->c_coll->coll_scatter((char *)tmp_buf,
                     block_size,
                     MPI_BYTE,
                     (char*)rbuf,
                     count,
                     dtype,
                     root_low_rank,
                     low_comm,
                     low_comm->c_coll->coll_scatter_module);

    if (low_rank == root_low_rank) {
        free(tmp_buf);
        tmp_buf = NULL;
    }
    if (reorder_buf != sbuf) {
        free(reorder_buf);
    }

    return OMPI_SUCCESS;

}

/* This function is called when the mca component han_scatter_handle_reorder_with_copy
 * is set to false and the topology is not mapbycore
 * It sets types_sent for a correct scatterw call
 * with smart datatypes that find the data and orders it
 */
int mca_coll_han_scatter_smartdtypes(int topo_lvl,
                                     int true_top_topo_lvl,
                                     const long int *send_sizes,
                                     int scount,
                                     ptrdiff_t block_extent,
                                     const struct ompi_datatype_t *sdtype,
                                     struct ompi_datatype_t ***types_sent,
                                     int w_rank,
                                     const mca_coll_han_module_t *han_module)
{
    int err = OMPI_SUCCESS;
    int line;
    const int *topo_index_from_rank = han_module->topo_tree->topo_index_from_rank;
    const int *rank_from_topo_index = han_module->topo_tree->rank_from_topo_index;
    int nb_dtypes = ompi_comm_size(han_module->sub_comm[topo_lvl]);

    /* We make 1 datatype of length scount * extent per task
     * So it's the number of 'child' tasks blocks
     * send_sizes[topo_lvl] = scount * number of 'child' tasks
     */
    size_t dtype_size;
    ompi_datatype_type_size(sdtype, &dtype_size);
    size_t nb_blocks = send_sizes[topo_lvl] / scount / dtype_size;
    int dtype_id = 0;

    /* new_types contains all the new types made and are returned out of the
     * function via types_sent.
     */
    struct ompi_datatype_t **new_types;
    ptrdiff_t *disps = NULL;
    int *sub_ranks = NULL;
    MALLOC(new_types, sizeof(struct ompi_datatype_t *) * nb_dtypes);
    MALLOC(disps, sizeof(ptrdiff_t) * nb_blocks);
    MALLOC(sub_ranks, sizeof(int) * (NB_TOPO_LVL-1));

    const int *origin = mca_coll_han_get_sub_ranks(han_module, w_rank);

    /* In case root != 0 we put 0s after topo_lvl to target the correct tasks
     * Example: 2 nodes, 2 sockets per node, 2 ranks per socket, distributed as follow
     *
     * Node 0                 Node 1
     * Socket 0   Socket 1    Socket 0   Socket 1
     * Rank 0     Rank 3      Rank 1     Rank 2
     * Rank 5     Rank 6      Rank 4     Rank 7
     *
     * with root chosen as rank 6
     * so in this scenario subranks are as follows
     * Rank 0 : [0 0 0]
     * Rank 6 : [0 1 1]
     * Rank 1 : [1 0 0]
     * Rank 7 : [1 1 1]
     *
     * on topo_lvl = INTER_NODE (2)
     * rank 6 will send message to rank 7 for ranks [1,2,4,7]
     * which means we take  [1 1 1]
     *                       | ______ we put 0s when topo < topo_lvl
     * topo_lvl = INTER_NODE | | |
     *                      [1 0 0]
     * so we selected [1 0 0] which is rank 1
     *
     * the layout of topo_index_from_rank will be [X,X,X,1,4,2,7]
     * which means that :
     *  ¤ rank_from_topo_index[topo_index_from_rank[4] + 0] = 1
     *  ¤ rank_from_topo_index[topo_index_from_rank[4] + 1] = 4
     *  ¤ rank_from_topo_index[topo_index_from_rank[4] + 2] = 2
     *  ¤ rank_from_topo_index[topo_index_from_rank[4] + 3] = 7
     */
    int topo;
    for (topo = 0; topo < topo_lvl; topo++) {
        sub_ranks[topo] = 0;
    }
    for (; topo < NB_TOPO_LVL-1; topo++) {
        sub_ranks[topo] = origin[topo];
    }

    for (dtype_id = 0; dtype_id < nb_dtypes; dtype_id++) {
        if (origin[topo_lvl] == dtype_id) {
            new_types[dtype_id] = NULL;
            continue;
        }
        sub_ranks[topo_lvl] = dtype_id;

        const int proc = topo_index_from_rank[mca_coll_han_get_global_rank(han_module, sub_ranks)];
        for (size_t block_id = 0; block_id < nb_blocks; block_id++) {
            int id = rank_from_topo_index[proc + block_id];
            disps[block_id] = id * scount * block_extent;
        }
        ompi_datatype_create_hindexed_block((int) nb_blocks, scount, disps, sdtype, &(new_types[dtype_id]));
        ompi_datatype_commit(&(new_types[dtype_id]));
    }

    *types_sent = new_types;
end:
    free(disps);
    free(sub_ranks);
    if (OMPI_SUCCESS != err) {
        char msg[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, msg, &len);
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                    "%s:%4d\tError %d occurred : %s", __FILE__, line, err, msg));

        for (int already_malloced = dtype_id; already_malloced >= 0; already_malloced--) {
            ompi_datatype_destroy(&(new_types[already_malloced]));
        }
        free(new_types);
    }
    return err;
}

/* ARG OUT : buff_sent, type_sent, buff_recv, type_recv
 * ARG INOUT : need_offset, sender_offset, internal_buff
 * ARG IN : everything else
 *
 * This function is giving the correct arguments for the scatter call
 * on the topological level topo_lvl
 * Since this part was a bit complex it was best to separate from the
 * recursive scatter.
 * The ARGS OUT change if : - if the process is root or acts like one in a scatter
 *                          - if we scatter on the first/last topo_lvl or common case
 *                          - if the process distribution is mapbycore
 *                          - if the process already acted as root in the scatter
 *                                  of the previous topo_lvl which compels us to
 *                                  add an offet to select the legitimate data
 *
 * The 4 OUT arguments below are the argument used in the scatter call or the recursive algorithm
 *     buff_sent is the address of the send buffer
 *     type_sent is the type of sent data
 *     buff_sent is the address of the recv buffer
 *     type_sent is the type of recv data
 *
 * need_offset marks if process already acted as root in the scatter of the previous topo_lvl
 * sender_offset is the offset to be added to internal_buff if need_offset == true
 * internal_buff is the internal buffer of data (either allocated or sbuf)
 * all ARG INOUT values are changed only in this function but initialized before
 *
 * w_rank World rank of the current task
 * root World rank of the root task
 * me My subranks
 * origin Subranks of the root
 * true_bot_topo_lvl The lowest level that has a communicator of size > 1
 * topo_lvl Current topological level the algorithm is in currently
 * last_lvl Last valid topo_lvl, useful if tasks aren't split according to every topo lvl
 * block_extent Number of bytes to add to find the address of the next value
 * send_sizes Sizes in bytes of the messages to be sent
 * rbuf Original rbuf provided to the recursive function
 * sdtype Original sdtype provided to the recursive function
 * rdtype Original rdtype provided to the recursive function
 * is_mapbycore A boolean signifying is the topology is mapbycore - see coll_han.h
 */
static void mca_coll_han_scatter_arguments(const int w_rank, const int root,
                                           const int *me, const int *origin,
                                           const int true_bot_topo_lvl,
                                           int topo_lvl, int last_lvl,
                                           bool *need_offset,
                                           size_t *sender_offset,
                                           const size_t block_extent,
                                           const size_t *send_sizes,
                                           void **buff_sent, void **buff_recv,
                                           void *rbuf,
                                           void **internal_buff,
                                           struct ompi_datatype_t *sdtype,
                                           struct ompi_datatype_t **type_sent,
                                           struct ompi_datatype_t *rdtype,
                                           struct ompi_datatype_t **type_recv,
                                           bool is_mapbycore,
                                           bool dtype_reorder)
{
    /* If I need smart datatypes to 'order' data and I'm root */
    if (dtype_reorder && !is_mapbycore && w_rank == root) {
        if (true_bot_topo_lvl == topo_lvl) {
            (*buff_sent) = (*internal_buff);
            (*buff_recv) = rbuf;
            (*type_sent) = sdtype;
            (*type_recv) = rdtype;
        } else {
            (*buff_sent) = (*internal_buff);
            (*buff_recv) = MPI_IN_PLACE;
        }

    /* If I'm receiver */
    } else if (me[topo_lvl] != origin[topo_lvl]) {
        if (true_bot_topo_lvl == topo_lvl) {
            (*buff_recv) = rbuf;
            (*type_recv) = rdtype;
        } else {
            (*buff_recv) = (*internal_buff);
            (*type_recv) = MPI_BYTE;
        }

    /* If I am one of the senders */
    } else {
        /* Here starts the tricky part
         * If you are not root in this topo_lvl it's simple
         * If you are root in this topo_lvl :
         *   - if you are mapbycore :
         *       care offset on roots after first comm + type_sent = sdtype
         *   - if you are NOT mapbycores :
         *       internal_buffer was reordered before
         *       so everything is in MPI_BYTE so we have
         *       dtype_size in offset and count
         */

        (*buff_sent) = (*internal_buff);
        /* root keeps sdtype for scatters to help for non-contiguous data */
        (*type_sent) = (w_rank == root && is_mapbycore)? sdtype:MPI_BYTE;
        /* If we dodged the intermediary buffer malloc for root */

        if (*need_offset) {
            /** Sbuf offset:
             * We keep sbuf to avoid doing too much scatters so we need
             * an offset on sbuf to keep sending appopriate data
             *
             * That offset is the previous message size (that is the
             * buff size we should have if we did scatter)
             * time the sub_rank of previous topo_lvl because that
             * selects the same buffer scatter would have selected last
             * topo_lvl. We add that value to previous offset value.
             */
            if (w_rank == root && is_mapbycore) {
                (*sender_offset) += send_sizes[last_lvl] * block_extent * me[last_lvl];
            } else {
                (*sender_offset) += send_sizes[last_lvl] * me[last_lvl];
            }

            (*buff_sent) += (*sender_offset);
        }

        if (true_bot_topo_lvl == topo_lvl) {
            (*buff_recv) = rbuf;
            (*type_recv) = rdtype;
        } else {
            (*buff_recv) = MPI_IN_PLACE;
        }
        /* If we reached here we're being root for the first time so
         * next time the data will need an offset since we do IN_PLACE */
        (*need_offset) = true;
    }
}

/* Reorder function that takes sbuf and put the data in order in  internal_buff */
static int mca_coll_han_scatter_reorder(void **internal_buff,
                                        int true_top_topo_lvl,
                                        size_t byte_count,
                                        ptrdiff_t block_extent,
                                        size_t *send_sizes,
                                        const void *sbuf, int scount,
                                        struct ompi_datatype_t *sdtype,
                                        struct ompi_datatype_t **type_sent,
                                        struct ompi_communicator_t *comm,
                                        const mca_coll_han_module_t *han_module)
{
    int err = OMPI_SUCCESS;
    int w_size = ompi_comm_size(comm);
    /* Data must be copied, let's be efficient packing it */
    /* Need to alloc buffer for reordering */
    if (((*internal_buff) = malloc(byte_count * w_size)) == NULL) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    send_sizes[true_top_topo_lvl] = byte_count;

    /** Reorder and packing:
     * Suppose, the message is 0 1 2 3 4 5 6 7 but the processes are
     * mapped on 2 nodes, for example |0 2 4 6| |1 3 5 7|. The messages to
     * leaders must be 0 2 4 6 and 1 3 5 7.
     * So the upper scatter must send 0 2 4 6 1 3 5 7.
     * The order is stored in the table 'topo_index_from_rank'
     */

    /* Reorder index */
    const int *index = han_module->topo_tree->topo_index_from_rank;
    size_t dtype_size;
    ompi_datatype_type_size(sdtype, &dtype_size);
    size_t length = dtype_size * scount;

    for (int process = 0; process < w_size; process++) {
        err = ompi_datatype_sndrcv((char*) sbuf + block_extent*process*scount,
                scount,
                sdtype,
                (*internal_buff) + length*index[process],
                (int) length,
                MPI_BYTE);
    }
    /* After reorder we have bytes not sdtype data */
    (*type_sent) = MPI_BYTE;
    return err;
}

/* Simple Scatter multi level algorithm
 * Consecutive scatters from NB_TOPO_LVL-2 to LEAF_LEVEL
 */
int mca_coll_han_scatter_intra_recursive(const void *sbuf, int scount,
                                         struct ompi_datatype_t *sdtype,
                                         void *rbuf, int rcount,
                                         struct ompi_datatype_t *rdtype,
                                         int root,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    int err = OMPI_SUCCESS;
    int line;
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;
    /* create the subcommunicators */
    if (OMPI_SUCCESS != mca_coll_han_comm_create_multi_level(comm, han_module)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                        "han cannot handle scatter within this communicator."
                        " Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype,
                                          rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    /* Determine if the number of processes per node is unbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
            "han cannot handle scatter with this communicator. It needs to fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype,
                                          rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    int w_rank = ompi_comm_rank(comm);
    /* get the sub_ranks of root for each topo level */
    const int *origin = mca_coll_han_get_sub_ranks(han_module, root);
    /* get my sub_ranks for each topo level */
    const int *me = mca_coll_han_get_sub_ranks(han_module, w_rank);
    /* Sub_comm alias */
    struct ompi_communicator_t **sub_comm = han_module->sub_comm;

    const void *buff_sent;
    void *buff_recv;
    void *internal_buff = NULL;
    size_t *send_sizes = NULL;
    size_t *recv_sizes = NULL;
    int *ones = NULL;
    int *zeroes = NULL;
    int *subranks = NULL;
    struct ompi_datatype_t *type_sent;
    struct ompi_datatype_t *type_recv;

    int count;
    const struct ompi_datatype_t *dtype;
    if (w_rank == root) {
        dtype = sdtype;
        count = scount;
    } else {
        dtype = rdtype;
        count = rcount;
    }

    /* We compute the future send and receive messages sizes in tabs */
    int true_top_topo_lvl = NB_TOPO_LVL-2;
    for (; true_top_topo_lvl > LEAF_LEVEL &&
            1 == ompi_comm_size(sub_comm[true_top_topo_lvl]);
            true_top_topo_lvl--); /* Empty loop */

    int true_bot_topo_lvl = LEAF_LEVEL;
    for (; true_bot_topo_lvl < true_top_topo_lvl &&
            1 == ompi_comm_size(sub_comm[true_bot_topo_lvl]);
            true_bot_topo_lvl++); /* Empty loop */

    MALLOC(send_sizes, sizeof(size_t) * (true_top_topo_lvl+1));
    MALLOC(recv_sizes, sizeof(size_t) * (true_top_topo_lvl+1));

    size_t byte_count = count;
    size_t dtype_size;
    ompi_datatype_type_size(dtype, &dtype_size);
    int first_scatter_lvl;
    /* We compute the future send and receive messages sizes in tabs */
    for (first_scatter_lvl = true_bot_topo_lvl; first_scatter_lvl < true_top_topo_lvl+1;
                                                first_scatter_lvl++) {
        if (w_rank == root && han_module->is_mapbycore) {
            send_sizes[first_scatter_lvl] = byte_count;
            recv_sizes[first_scatter_lvl] = byte_count;
        } else {
            send_sizes[first_scatter_lvl] = byte_count * dtype_size;
            recv_sizes[first_scatter_lvl] = byte_count * dtype_size;
        }
        if (me[first_scatter_lvl] != origin[first_scatter_lvl] ||
                            true_top_topo_lvl == first_scatter_lvl) {
            break;
        }
        byte_count *= ompi_comm_size(sub_comm[first_scatter_lvl]);
    }

    int is_scatter_with_root = true;
    if (first_scatter_lvl != true_top_topo_lvl) {
        for (int topo = true_top_topo_lvl; topo > first_scatter_lvl; topo--) {
            if (me[topo] != origin[topo]) {
                is_scatter_with_root = false;
            }
        }
    }

    recv_sizes[true_bot_topo_lvl] = count;
    byte_count *= dtype_size;

    ptrdiff_t block_extent;
    ompi_datatype_type_extent(dtype, &block_extent);

    int size = ompi_comm_size(comm);
    bool dtype_reorder = !mca_coll_han_component.han_scatter_handle_reorder_with_copy;
    if (w_rank != root) {
        /* We allocate 1 buffer with size
         * count * dtype_size * (comm_size[first_scatter_lvl-1] * ... * comm_size[LEAF_LEVEL])
         */
        if (first_scatter_lvl != true_bot_topo_lvl) {
            MALLOC(internal_buff, byte_count);
        }

    } else {
        if (han_module->is_mapbycore) {
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                "[%d]: Han scatter: no need to reorder :", ompi_comm_rank(comm)));
            internal_buff = (void *) sbuf;
        } else {
            if (dtype_reorder) {
                OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                    "[%d]: Han scatter: need smart datatypes :", ompi_comm_rank(comm)));

                MALLOC(ones, sizeof(int) * size);
                MALLOC(zeroes, sizeof(int) * size);
                internal_buff = (void *) sbuf;
                for (int task = 0; task < size; task++) {
                    ones[task] = 1;
                    zeroes[task] = 0;
                }
            } else {
                OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                    "[%d]: Han scatter: needs reordering: ", ompi_comm_rank(comm)));

                err = mca_coll_han_scatter_reorder(&internal_buff,
                                                   true_top_topo_lvl,
                                                   byte_count,
                                                   block_extent,
                                                   send_sizes,
                                                   sbuf, scount,
                                                   sdtype,
                                                   &type_sent,
                                                   comm,
                                                   han_module);
                if (OMPI_SUCCESS != err) {
                    goto end;
                }
            }
        }
    }
    bool need_offset = false;
    size_t sender_offset = 0;

    /* Recursive algorithm: from higher to lower */
    int last_lvl = first_scatter_lvl;
    for (int topo_lvl = first_scatter_lvl; topo_lvl >= true_bot_topo_lvl; topo_lvl--) {
        if (ompi_comm_size(sub_comm[topo_lvl]) < 2) {
            continue;
        }

        /* Once an MPI process owns data, it is a scatter root on remaining levels.
         * MPI processes do not perform scatter on a specific level if
         * no MPI processes they share this sub_communicator with is root
         * If we are considered one of the roots in the communication
         */
        mca_coll_han_scatter_arguments(w_rank, root,
                                       me, origin,
                                       true_bot_topo_lvl,
                                       topo_lvl, last_lvl,
                                       &need_offset,
                                       &sender_offset,
                                       block_extent,
                                       send_sizes,
                                       (void **) &buff_sent, &buff_recv,
                                       rbuf,
                                       &internal_buff,
                                       sdtype, &type_sent,
                                       rdtype, &type_recv,
                                       han_module->is_mapbycore,
                                       dtype_reorder);

        if (!dtype_reorder || han_module->is_mapbycore || !is_scatter_with_root ) {
            /* Classical scatter */
            sub_comm[topo_lvl]->c_coll->coll_scatter(buff_sent,
                                           (int) send_sizes[topo_lvl],
                                           type_sent,
                                           buff_recv,
                                           (int) recv_sizes[topo_lvl],
                                           type_recv,
                                           origin[topo_lvl],
                                           sub_comm[topo_lvl],
                                           sub_comm[topo_lvl]->c_coll->coll_scatter_module);

        } else if (topo_lvl > true_bot_topo_lvl) {
            /* Root rank builds hindexed datatype to reorder for all data
             * (without local copy)
             * Other processes won't have to reorder and may scatter as is
             */
            struct ompi_datatype_t **types_sent;
            if (w_rank == root) {
                err = mca_coll_han_scatter_smartdtypes(topo_lvl,
                                                       true_top_topo_lvl,
                                                       send_sizes,
                                                       scount,
                                                       block_extent,
                                                       sdtype,
                                                       &types_sent,
                                                       w_rank,
                                                       han_module);
                CHECK(err);
            }
            err = sub_comm[topo_lvl]->c_coll->coll_scatterw(buff_sent,
                                            ones,
                                            zeroes,
                                            types_sent,
                                            buff_recv,
                                            (int) recv_sizes[topo_lvl],
                                            type_recv,
                                            origin[topo_lvl],
                                            sub_comm[topo_lvl],
                                            sub_comm[topo_lvl]->c_coll->coll_scatterw_module);

            CHECK(err);
            /* Freeing what was allocated inside the smartdtypes function */
            if (w_rank == root) {
                int nb_dtypes = ompi_comm_size(sub_comm[topo_lvl]);
                for (int dtype_id = 0; dtype_id < nb_dtypes; dtype_id++) {
                    if (types_sent[dtype_id] != NULL) {
                        ompi_datatype_destroy(&(types_sent[dtype_id]));
                    }
                }
                free(types_sent);
            } else {
                is_scatter_with_root = false;
            }
        } else {
            /* Last scatter doesn't need to build datatypes because we do not
             * to agregate buffers to be sent in the next topological level
             * A scatterv therefore suffice if displacement is good
             */
            int lowest_size = ompi_comm_size(sub_comm[topo_lvl]);
            MALLOC(subranks, sizeof(int) * (NB_TOPO_LVL-1));
            int *counts_send = ones; /* keep the same buffer */
            int *displacements = zeroes; /* keep the same buffer */

            if (w_rank == root) {
                int topo;
                for (topo = 0; topo < topo_lvl; topo++) {
                    subranks[topo] = 0;
                }
                for (; topo < NB_TOPO_LVL-1; topo++) {
                    subranks[topo] = origin[topo];
                }

                for (int task = 0; task < lowest_size; task++) {
                    subranks[topo_lvl] = task;
                    counts_send[task] = scount;
                    displacements[task] =
                        mca_coll_han_get_global_rank(han_module, subranks) * scount;
                }
            }

            free(subranks);
            err = sub_comm[topo_lvl]->c_coll->coll_scatterv(buff_sent,
                                            counts_send,
                                            displacements,
                                            type_sent,
                                            buff_recv,
                                            (int) recv_sizes[topo_lvl],
                                            type_recv,
                                            origin[topo_lvl],
                                            sub_comm[topo_lvl],
                                            sub_comm[topo_lvl]->c_coll->coll_scatterv_module);

            CHECK(err);
        }
        last_lvl = topo_lvl;
    }

end:
    if (sbuf != internal_buff) {
        free(internal_buff);
    }
    if (dtype_reorder && w_rank == root && !han_module->is_mapbycore) {
            free(ones);
            free(zeroes);
    }

    if (err) {
        char msg[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, msg, &len);
        opal_output_verbose(3, mca_coll_han_component.han_output,
                            (ompi_coll_base_framework.framework_output,
                             "%s:%4d\tError %d occurred : %s", __FILE__, line, err, msg));
    }

    free(send_sizes);
    free(recv_sizes);
    return err;
}

/* N-level scatter with a pipeline of data for each level
 * At LEAF_LEVEL of when we can fill only one send in the pipeline
 * we switch to a classic scatter algorithm
 */
int mca_coll_han_scatter_intra_pipeline(const void *sbuf, int scount,
                                        struct ompi_datatype_t *sdtype,
                                        void *rbuf, int rcount,
                                        struct ompi_datatype_t *rdtype,
                                        int root,
                                        struct ompi_communicator_t *comm,
                                        mca_coll_base_module_t *module)
{
    int err = OMPI_SUCCESS;
    int line;
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;
    /* create the subcommunicators */
    if (OMPI_SUCCESS != mca_coll_han_comm_create_multi_level(comm, han_module)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                        "han cannot handle scatter within this communicator."
                        " Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype,
                                          rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    /* Determine if the number of processes per node is unbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
            "han cannot handle scatter with this communicator. It needs to fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype,
                                          rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    int w_rank = ompi_comm_rank(comm);
    /* get the sub_ranks of root for each topo level */
    const int *origin = mca_coll_han_get_sub_ranks(han_module, root);
    /* get my sub_ranks for each topo level */
    const int *me = mca_coll_han_get_sub_ranks(han_module, w_rank);

    /* Sub_comm alias */
    struct ompi_communicator_t **sub_comm = han_module->sub_comm;

    const void *buff_sent;
    void *buff_recv;
    struct ompi_datatype_t *type_sent;
    struct ompi_datatype_t *type_recv;

    int count;
    const struct ompi_datatype_t *dtype;
    if (w_rank == root) {
        dtype = sdtype;
        count = scount;
    } else {
        dtype = rdtype;
        count = rcount;
    }

    /* We compute the future send and receive messages sizes in tabs */
    int true_top_topo_lvl = NB_TOPO_LVL-2;
    for (; true_top_topo_lvl > LEAF_LEVEL &&
            1 == ompi_comm_size(sub_comm[true_top_topo_lvl]);
            true_top_topo_lvl--); /* Empty loop */

    int true_bot_topo_lvl = LEAF_LEVEL;
    for (; true_bot_topo_lvl < true_top_topo_lvl &&
            1 == ompi_comm_size(sub_comm[true_bot_topo_lvl]);
            true_bot_topo_lvl++); /* Empty loop */

    void *internal_buff = NULL;
    ompi_request_t **requests = NULL;
    ompi_status_public_t *statuses = NULL;
    size_t *send_sizes = NULL;
    MALLOC(send_sizes, sizeof(size_t) * (true_top_topo_lvl+1));

    size_t dtype_size;
    ompi_datatype_type_size(dtype, &dtype_size);
    size_t byte_count = count * dtype_size;
    /* We compute the future send and receive messages sizes in tabs */
    int first_scatter_lvl;
    for (first_scatter_lvl = true_bot_topo_lvl; first_scatter_lvl < true_top_topo_lvl+1;
                                                first_scatter_lvl++) {
        send_sizes[first_scatter_lvl] = byte_count;
        if (me[first_scatter_lvl] != origin[first_scatter_lvl] ||
                            true_top_topo_lvl == first_scatter_lvl) {
            break;
        }

        byte_count *= ompi_comm_size(sub_comm[first_scatter_lvl]);
    }

    for (int scatter_lvl = first_scatter_lvl+1; scatter_lvl < true_top_topo_lvl+1;
                                                scatter_lvl++) {
        send_sizes[scatter_lvl] = send_sizes[scatter_lvl-1]
                                  * ompi_comm_size(sub_comm[scatter_lvl]);
    }

    ptrdiff_t block_extent;
    ompi_datatype_type_extent(dtype, &block_extent);

    if (w_rank == root) {
        int contiguous = ompi_datatype_is_contiguous_memory_layout(dtype, count);
        if (!contiguous || !han_module->is_mapbycore) {
            /* If reorder is needed, we do it */
            err = mca_coll_han_scatter_reorder(&internal_buff,
                                               true_top_topo_lvl,
                                               byte_count,
                                               block_extent,
                                               send_sizes,
                                               sbuf, scount,
                                               sdtype,
                                               &type_sent,
                                               comm,
                                               han_module);
            CHECK(err);
        } else {
            internal_buff = (void *) sbuf;
        }

    } else {
        /* We allocate 1 buffer with size
         * count * dtype_size * (comm_size[first_scatter_lvl-1] * ... * comm_size[LEAF_LEVEL])
         */
        if (first_scatter_lvl != true_bot_topo_lvl) {
            MALLOC(internal_buff, byte_count);
        }
    }

    int segment_min_count = mca_coll_han_component.han_scatter_segsize;
    if (segment_min_count == 0) {
        segment_min_count = 1;
    }
    int nb_segments = mca_coll_han_component.han_scatter_max_nb_segments;
    if (nb_segments == 0) {
        nb_segments = 1;
    }
    /* max number of segments we allow */
    MALLOC(requests, (nb_segments+1) * sizeof(ompi_request_t*));
    int nb_requests = 0;

    bool need_offset = false;
    size_t sender_offset = 0;
    int last_lvl = first_scatter_lvl;
    /* Recursive algorithm: from higher to lower */
    for (int topo_lvl = first_scatter_lvl; topo_lvl >= true_bot_topo_lvl; topo_lvl--) {
        if (ompi_comm_size(sub_comm[topo_lvl]) < 2) {
            continue;
        }
        /* To be noted we compute here how many segment we'd like to use however
         * we calculate it with the total amount of data sent by one sender
         * where under this we compute size_per_segment which is computed with
         * the total amount of data / number of subranks
         */
        int expected_nb_segment = send_sizes[topo_lvl] / segment_min_count;
        if (expected_nb_segment < nb_segments) {
            if (expected_nb_segment < 1) {
                nb_segments = 1;
            } else {
                nb_segments = expected_nb_segment;
            }
        }

        mca_coll_han_scatter_arguments(w_rank, root,
                                       me, origin,
                                       true_bot_topo_lvl,
                                       topo_lvl, last_lvl,
                                       &need_offset,
                                       &sender_offset,
                                       1,
                                       send_sizes,
                                       (void **) &buff_sent, &buff_recv,
                                       rbuf,
                                       &internal_buff,
                                       sdtype, &type_sent,
                                       rdtype, &type_recv,
                                       han_module->is_mapbycore,
                                       0);

        if (1 == nb_segments || true_bot_topo_lvl == topo_lvl) {
            err = ompi_request_wait_all(nb_requests, requests, MPI_STATUSES_IGNORE);
            CHECK(err);
            /* Classical scatter */
            err = sub_comm[topo_lvl]->c_coll->coll_scatter(buff_sent,
                                       send_sizes[topo_lvl],
                                       MPI_BYTE,
                                       buff_recv,
                                       (topo_lvl != true_bot_topo_lvl)?
                                           (int) send_sizes[topo_lvl]:rcount,
                                       type_recv,
                                       origin[topo_lvl],
                                       sub_comm[topo_lvl],
                                       sub_comm[topo_lvl]->c_coll->coll_scatter_module);
            CHECK(err);

        } else {
            /* The low_size is the number of (low) ranks the current receiver will
             * have to distribute data to during scatters on incoming
             * topological levels
             */
            int low_size = 1;
            for (int i = topo_lvl-1; i >= LEAF_LEVEL; i--) {
                low_size *= ompi_comm_size(sub_comm[i]);
            }
            /* size_per_low_rank is the total amount of data to send to one rank */
            int size_per_low_rank = count * dtype_size;
            /* size_per_segment is the amount of data we send to one rank per segment */
            int size_per_segment = size_per_low_rank / nb_segments;
            /* Prepare the vector for up scatter where we'll distribute the
             * total memory (send_sizes[topo_lvl]) equally for each low rank
             * We have :
             *     ¤ low_size blocks (see previous commentary)
             *     ¤ size_per_segment block size (it is what we send per segment)
             *          if it is too much we overwrite the datatype correct size
             *     ¤ size_per_low_rank the stride cuz it is the total data for each block
             */
            ompi_datatype_t *vector = NULL;
            ompi_datatype_create_vector(low_size, size_per_segment, size_per_low_rank,
                                        MPI_BYTE, &vector);
            ompi_datatype_commit(&vector);
            opal_datatype_resize(&vector->super, 0, send_sizes[topo_lvl]);

            /* We pushed this off as late as possible */
            err = ompi_request_wait_all(nb_requests, requests, MPI_STATUSES_IGNORE);
            if (err != OMPI_SUCCESS) {
                ompi_datatype_destroy(&vector);
                goto end;
            }

            /* Actualize the number of requests for next level */
            nb_requests = nb_segments;

            for (int segment = 0; segment < nb_segments; segment++) {
                /* size of buffer to be sent this iteration */
                int to_be_sent = (int) (size_per_low_rank - size_per_segment * segment);
                if (to_be_sent > size_per_segment) {
                    to_be_sent = size_per_segment;
                }

                /* Datatype done again if we send too much data (last iteration) */
                if (segment == nb_segments-1) {
                    int surplus = size_per_low_rank % nb_segments;
                    ompi_datatype_destroy(&vector);
                    ompi_datatype_create_vector(low_size, to_be_sent + surplus,
                                                size_per_low_rank, MPI_BYTE, &vector);
                    ompi_datatype_commit(&vector);
                    opal_datatype_resize(&vector->super, 0, send_sizes[topo_lvl]);
                }

                int offset = size_per_segment * segment;
                if (buff_recv != MPI_IN_PLACE && buff_recv != rbuf) {
                    buff_recv = internal_buff + offset;
                }

                err = sub_comm[topo_lvl]->c_coll->coll_iscatter(
                                       buff_sent + offset,
                                       1,
                                       vector,
                                       buff_recv,
                                       1,
                                       vector,
                                       origin[topo_lvl],
                                       sub_comm[topo_lvl],
                                       &requests[segment],
                                       sub_comm[topo_lvl]->c_coll->coll_iscatter_module);
                if (err != OMPI_SUCCESS) {
                    ompi_datatype_destroy(&vector);
                    goto end;
                }
            }
            ompi_datatype_destroy(&vector);
        }
        last_lvl = topo_lvl;
    }

end:
    free(send_sizes);
    free(requests);
    if (sbuf != internal_buff) {
        free(internal_buff);
    }

    if (err) {
        char msg[MPI_MAX_ERROR_STRING];
        int len;
        MPI_Error_string(err, msg, &len);
        opal_output_verbose(3, mca_coll_han_component.han_output,
                            (ompi_coll_base_framework.framework_output,
                             "%s:%4d\tError %d occurred : %s", __FILE__, line, err, msg));
    }

    return err;
}

/* Scatter algorithm for 2 levels where we try not to go beyond cache memory
 * because it causes a spike in latency to do some swap.
 * Carefully picking the memory limit is essential. Needing a reordering of
 * data kind of defeat the purpose of the algorithm
 *
 * If X = count * dtype_size, and N the number of nodes and ppn the number of tasks per node
 *     then the user has at minimum already allocated X bytes for each tasks
 *     except on root he has reserved : N*ppn*X+X bytes (sbuf+rbuf)
 * If we define M as the chosen memory limit.
 *     The algorithm has reserved M bytes for intermediary buffer
 *     and needs ppn*X/M rounds to finish
 * You should chose M such as max_cache_memory/2-X >= M
 *
 * Top level scatter collective is splitted in multiple ones, each sends data for all
 * final ranks but just enough that intermediary process buffers fit in caches.
 */
int
mca_coll_han_scatter_intra_memcare(const void *sbuf, int scount,
                                   struct ompi_datatype_t *sdtype,
                                   void *rbuf, int rcount,
                                   struct ompi_datatype_t *rdtype,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t * module)
{
    int err = OMPI_SUCCESS;
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *) module;

    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator (not 2 levels). Fall back on another component\n");
        /* HAN cannot work with this communicator so fallback scatter collective */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                          comm, comm->c_coll->coll_scatter_module);
    }

    /* create the subcommunicators */
    if (OMPI_SUCCESS != mca_coll_han_comm_create_multi_level(comm, han_module)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle collectives within this communicator."
                             " Fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle scatter with this communicator. It needs to fall back on another component\n"));
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, scatter);
        return comm->c_coll->coll_scatter(sbuf, scount, sdtype, rbuf, rcount, rdtype, root,
                                          comm, han_module->previous_scatter_module);
    }

    int w_rank = ompi_comm_rank(comm);
    int w_size = ompi_comm_size(comm);
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int low_size = ompi_comm_size(low_comm);
    int segsize = mca_coll_han_component.han_scatter_memsize;

    int count;
    const struct ompi_datatype_t *dtype;
    if (w_rank == root) {
        dtype = sdtype;
        count = scount;
    } else {
        dtype = rdtype;
        count = rcount;
    }

    size_t dtype_size;
    size_t block_size;
    ompi_datatype_type_size(dtype, &dtype_size);
    block_size = dtype_size * count;

    void *recv_buff;
    int recv_contg = 1;
    if (rbuf != MPI_IN_PLACE) {
        recv_contg = ompi_datatype_is_contiguous_memory_layout(rdtype, rcount);
    }

    ompi_datatype_t *vector = NULL;
    char *internal_buf = NULL;
    char *reorder_buf = NULL;  /* allocated memory */

    /* If the rdtype is not contiguous at leaf_level we need to do this to translate */
    if (!recv_contg) {
        MALLOC(recv_buff, block_size);
    } else {
        recv_buff = rbuf;
    }

    if (w_rank == root) {
        int is_contiguous = ompi_datatype_is_contiguous_memory_layout(sdtype, scount);

        /* Datatype sizes may differ on each rank. To get easy framgentation
         * whatever the datatypes, all is converted into MPI_BYTE and therefore
         * data is either already contiguous or packed */
        if (han_module->is_mapbycore && is_contiguous) {
            /* The copy of the data is avoided */
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                                 "[%d]: Han scatter: no need to reorder: ", w_rank));
            reorder_buf = (char *)sbuf;     //NOSONAR
        } else {
            /* Data must be copied, let's be efficient packing it */
            OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                        "[%d]: Han scatter: needs reordering or compacting: ", w_rank));

            MALLOC(reorder_buf, block_size * w_size);

            /** Reorder and packing:
             * Suppose, the message is 0 1 2 3 4 5 6 7 but the processes are
             * mapped on 2 nodes, for example |0 2 4 6| |1 3 5 7|. The messages to
             * leaders must be 0 2 4 6 and 1 3 5 7.
             * So the upper scatter must send 0 2 4 6 1 3 5 7.
             * In general, the topo[i*topolevel +1]  must be taken.
             */

            /* Reorder index */
            const int *index = han_module->topo_tree->topo_index_from_rank;

            size_t block_extent;
            ompi_datatype_type_extent(dtype, &block_extent);
            for (int process = 0; process < w_size; process++) {
                err = ompi_datatype_sndrcv(((char*)sbuf) + (ptrdiff_t)(block_extent * count * process),   //NOSONAR /* Discard const qualifier for sbuf */
                        scount,
                        sdtype,
                        reorder_buf + block_size*index[process],
                        (int) block_size,
                        MPI_BYTE);
            }
            if (OMPI_SUCCESS != err) {
                goto end;
            }
        }
    }

    /* Prepare for subranks manipulation */
    const int *origin = mca_coll_han_get_sub_ranks(han_module, root);
    const int *me = mca_coll_han_get_sub_ranks(han_module, w_rank);

    /* Distribute the total memory (segsize) equally for each task on low scatter */
    int size_per_low_task = segsize / low_size;
    int nb_segments = ((int) block_size / size_per_low_task);
    if (block_size % size_per_low_task) {
        nb_segments++;
    }

    /* Limit the size of memory to the memory specified by the user */
    if (me[LEAF_LEVEL] == origin[LEAF_LEVEL]) {
        if (nb_segments > 1) {
            MALLOC(internal_buf, segsize);
        } else {
            MALLOC(internal_buf, block_size * low_size);
        }
    }

    /* Prepare the datatype for up scatter */
    if (w_rank == root) {
        ompi_datatype_create_vector(low_size, size_per_low_task, (int) block_size,
                                    MPI_BYTE, &vector);
        ompi_datatype_commit(&vector);
        opal_datatype_resize(&vector->super, 0, dtype_size * count * low_size);
    }

    for (int segment = 0; segment < nb_segments; segment++) {
        /* size of buffer to be sent this iteration */
        int to_be_sent = (int) (block_size - size_per_low_task * segment);
        if (to_be_sent > size_per_low_task) {
            to_be_sent = size_per_low_task;
        }

        if (w_rank == root && to_be_sent < size_per_low_task) {
            /* Datatype need to be done again if it's the last iteration */
            ompi_datatype_destroy(&vector);
            ompi_datatype_create_vector(low_size, to_be_sent, (int) block_size,
                                        MPI_BYTE, &vector);
            ompi_datatype_commit(&vector);
            opal_datatype_resize(&vector->super, 0, dtype_size * count * low_size);
        }

        int offset = size_per_low_task * segment;
        if (me[LEAF_LEVEL] == origin[LEAF_LEVEL]) {
            err = up_comm->c_coll->coll_scatter(reorder_buf + offset,
                    1,
                    vector,
                    internal_buf,
                    to_be_sent * low_size,
                    MPI_BYTE,
                    origin[INTER_NODE],
                    up_comm,
                    up_comm->c_coll->coll_scatter_module);
        }
        if (OMPI_SUCCESS != err) {
            goto end;
        }

        /* 2. low scatter on nodes leaders */
        if (MPI_IN_PLACE == rbuf) {
            err = low_comm->c_coll->coll_scatter(internal_buf,
                    to_be_sent,
                    MPI_BYTE,
                    MPI_IN_PLACE,
                    to_be_sent,
                    rdtype,
                    origin[LEAF_LEVEL],
                    low_comm,
                    low_comm->c_coll->coll_scatter_module);
        } else {
            err = low_comm->c_coll->coll_scatter(internal_buf,
                    to_be_sent,
                    MPI_BYTE,
                    (char *) recv_buff + offset,
                    to_be_sent,
                    MPI_BYTE,
                    origin[LEAF_LEVEL],
                    low_comm,
                    low_comm->c_coll->coll_scatter_module);
        }
        if (OMPI_SUCCESS != err) {
            goto end;
        }
    }

    if (!recv_contg) {  /* bad use case if we enter in there */
        err = ompi_datatype_sndrcv((char*)recv_buff,//NOSONAR
                             (int) block_size,
                             MPI_BYTE,
                             rbuf,
                             rcount,
                             rdtype);

        if (OMPI_SUCCESS != err) {
           goto end;
        }
    }

end:
    free(internal_buf);
    if (w_rank == root) {
        if (sbuf != reorder_buf) {
            free(reorder_buf);
        }
        if (vector != NULL) {
            ompi_datatype_destroy(&vector);
        }
    }
    return err;
}

