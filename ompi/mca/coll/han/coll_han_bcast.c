/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020      Cisco Systems, Inc.  All rights reserved.
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
 * This files contains all the hierarchical implementations of bcast
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

static int mca_coll_han_bcast_t0_task(void *task_args);
static int mca_coll_han_bcast_t1_task(void *task_args);

static inline void
mca_coll_han_set_bcast_args(mca_coll_han_bcast_args_t * args, mca_coll_task_t * cur_task, void *buff,
                            int seg_count, struct ompi_datatype_t *dtype,
                            int root_up_rank, int root_low_rank,
                            struct ompi_communicator_t *up_comm,
                            struct ompi_communicator_t *low_comm,
                            int num_segments, int cur_seg, int w_rank, int last_seg_count,
                            bool noop)
{
    args->cur_task = cur_task;
    args->buff = buff;
    args->seg_count = seg_count;
    args->dtype = dtype;
    args->root_low_rank = root_low_rank;
    args->root_up_rank = root_up_rank;
    args->up_comm = up_comm;
    args->low_comm = low_comm;
    args->num_segments = num_segments;
    args->cur_seg = cur_seg;
    args->w_rank = w_rank;
    args->last_seg_count = last_seg_count;
    args->noop = noop;
}

/*
 * Each segment of the message needs to go though 2 steps to perform MPI_Bcast:
 *     ub: upper level (inter-node) bcast
 *     lb: low level (shared-memory or intra-node) bcast.
 * Hence, in each iteration, there is a combination of collective operations which is called a task.
 *        | seg 0 | seg 1 | seg 2 | seg 3 |
 * iter 0 |  ub   |       |       |       | task: t0, contains ub
 * iter 1 |  lb   |  ub   |       |       | task: t1, contains ub and lb
 * iter 2 |       |  lb   |  ub   |       | task: t1, contains ub and lb
 * iter 3 |       |       |  lb   |  ub   | task: t1, contains ub and lb
 * iter 4 |       |       |       |  lb   | task: t1, contains lb
 */
int
mca_coll_han_bcast_intra(void *buf,
                         int count,
                         struct ompi_datatype_t *dtype,
                         int root,
                         struct ompi_communicator_t *comm, mca_coll_base_module_t * module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err, seg_count = count, w_rank = ompi_comm_rank(comm);
    ompi_communicator_t *low_comm, *up_comm;
    ptrdiff_t extent, lb;
    size_t dtype_size;

    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(0, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_bcast(buf, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_bcast(buf, count, dtype, root,
                                          comm, han_module->previous_bcast_module);
    }

    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    mca_coll_han_topo_init(comm, han_module, 2);
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (imbalance). Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return han_module->previous_bcast(buf, count, dtype, root,
                                          comm, han_module->previous_bcast_module);
    }

    ompi_datatype_get_extent(dtype, &lb, &extent);
    ompi_datatype_type_size(dtype, &dtype_size);

    /* use MCA parameters for now */
    low_comm = han_module->cached_low_comms[mca_coll_han_component.han_bcast_low_module];
    up_comm = han_module->cached_up_comms[mca_coll_han_component.han_bcast_up_module];
    COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_han_component.han_bcast_segsize, dtype_size,
                                seg_count);

    int num_segments = (count + seg_count - 1) / seg_count;
    OPAL_OUTPUT_VERBOSE((20, mca_coll_han_component.han_output,
                         "In HAN seg_count %d count %d num_seg %d\n",
                         seg_count, count, num_segments));

    int *vranks = han_module->cached_vranks;
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);

    int root_low_rank, root_up_rank;
    mca_coll_han_get_ranks(vranks, root, low_size, &root_low_rank, &root_up_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d]: root_low_rank %d root_up_rank %d\n", w_rank, root_low_rank,
                         root_up_rank));

    /* Create t0 tasks for the first segment */
    mca_coll_task_t *t0 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t0 task arguments */
    mca_coll_han_bcast_args_t *t = malloc(sizeof(mca_coll_han_bcast_args_t));
    mca_coll_han_set_bcast_args(t, t0, (char *)buf, seg_count, dtype,
                                root_up_rank, root_low_rank, up_comm, low_comm,
                                num_segments, 0, w_rank, count - (num_segments - 1) * seg_count,
                                low_rank != root_low_rank);
    /* Init the first task */
    init_task(t0, mca_coll_han_bcast_t0_task, (void *) t);
    issue_task(t0);

    /* Create t1 task */
    mca_coll_task_t *t1 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t1 task arguments */
    t->cur_task = t1;
    /* Init the t1 task */
    init_task(t1, mca_coll_han_bcast_t1_task, (void *) t);
    issue_task(t1);

    while (t->cur_seg <= t->num_segments - 2) {
        /* Create t1 task */
        t->cur_task = t1 = OBJ_NEW(mca_coll_task_t);
        t->buff = (char *) t->buff + extent * seg_count;
        t->cur_seg = t->cur_seg + 1;
        /* Init the t1 task */
        init_task(t1, mca_coll_han_bcast_t1_task, (void *) t);
        issue_task(t1);
    }

    free(t);

    return OMPI_SUCCESS;
}

/* t0 task: issue and wait for the upper level ibcast of segment 0 */
int mca_coll_han_bcast_t0_task(void *task_args)
{
    mca_coll_han_bcast_args_t *t = (mca_coll_han_bcast_args_t *) task_args;

    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d]: in t0 %d\n", t->w_rank,
                         t->cur_seg));
    OBJ_RELEASE(t->cur_task);
    if (t->noop) {
        return OMPI_SUCCESS;
    }
    t->up_comm->c_coll->coll_bcast((char *) t->buff, t->seg_count, t->dtype, t->root_up_rank,
                                   t->up_comm, t->up_comm->c_coll->coll_bcast_module);
    return OMPI_SUCCESS;
}

/* t1 task:
 * 1. issue the upper level ibcast of segment cur_seg + 1
 * 2. issue the low level bcast of segment cur_seg
 * 3. wait for the completion of the ibcast
 */
int mca_coll_han_bcast_t1_task(void *task_args)
{
    mca_coll_han_bcast_args_t *t = (mca_coll_han_bcast_args_t *) task_args;
    ompi_request_t *ibcast_req = NULL;
    int tmp_count = t->seg_count;
    ptrdiff_t extent, lb;

    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output, "[%d]: in t1 %d\n", t->w_rank,
                         t->cur_seg));
    OBJ_RELEASE(t->cur_task);
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    if (!t->noop) {
        if (t->cur_seg <= t->num_segments - 2 ) {
            if (t->cur_seg == t->num_segments - 2) {
                tmp_count = t->last_seg_count;
            }
            t->up_comm->c_coll->coll_ibcast((char *) t->buff + extent * t->seg_count,
                                            tmp_count, t->dtype, t->root_up_rank,
                                            t->up_comm, &ibcast_req,
                                            t->up_comm->c_coll->coll_ibcast_module);
        }
    }

    /* are we the last segment to be pushed downstream ? */
    tmp_count = (t->cur_seg == (t->num_segments - 1)) ? t->last_seg_count : t->seg_count;
    t->low_comm->c_coll->coll_bcast((char *) t->buff,
                                    tmp_count, t->dtype, t->root_low_rank, t->low_comm,
                                    t->low_comm->c_coll->coll_bcast_module);

    if (NULL != ibcast_req) {
        ompi_request_wait(&ibcast_req, MPI_STATUS_IGNORE);
    }

    return OMPI_SUCCESS;
}

/*
 * Short implementation of bcast that only does hierarchical
 * communications without tasks.
 */
int
mca_coll_han_bcast_intra_simple(void *buf,
                                int count,
                                struct ompi_datatype_t *dtype,
                                int root,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    ompi_communicator_t *low_comm, *up_comm;
    int err;
#if OPAL_ENABLE_DEBUG
    int w_rank = ompi_comm_rank(comm);
#endif

    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(0, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_bcast(buf, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }
    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_new(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_bcast(buf, count, dtype, root,
                                          comm, han_module->previous_bcast_module);
    }
    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    mca_coll_han_topo_init(comm, han_module, 2);
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (imbalance). Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return han_module->previous_bcast(buf, count, dtype, root,
                                          comm, han_module->previous_bcast_module);
    }

    low_comm = han_module->sub_comm[LEAF_LEVEL];
    up_comm = han_module->sub_comm[INTER_NODE];

    int *vranks = han_module->cached_vranks;
    int low_rank = ompi_comm_rank(low_comm);
    int low_size = ompi_comm_size(low_comm);
    int root_low_rank, root_up_rank;

    mca_coll_han_get_ranks(vranks, root, low_size, &root_low_rank, &root_up_rank);
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d]: root_low_rank %d root_up_rank %d\n",
                         w_rank, root_low_rank, root_up_rank));

    if (low_rank == root_low_rank) {
        up_comm->c_coll->coll_bcast(buf, count, dtype, root_up_rank,
                                    up_comm, up_comm->c_coll->coll_bcast_module);

        /* To remove when han has better sub-module selection.
           For now switching to ibcast enables to make runs with libnbc. */
        //ompi_request_t req;
        //up_comm->c_coll->coll_ibcast(buf, count, dtype, root_up_rank,
        //                             up_comm, &req, up_comm->c_coll->coll_ibcast_module);
        //ompi_request_wait(&req, MPI_STATUS_IGNORE);

    }
    low_comm->c_coll->coll_bcast(buf, count, dtype, root_low_rank,
                                 low_comm, low_comm->c_coll->coll_bcast_module);

    return OMPI_SUCCESS;
}


int
mca_coll_han_bcast_intra_recursive(void *buff,
                                   int count,
                                   struct ompi_datatype_t *dtype,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module)
{
    /* create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err;
    int *target;
    int *me;
    int w_rank;
    ompi_communicator_t *sub_comm;

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_bcast(buff, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }

    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_bcast(buff, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }

    w_rank = ompi_comm_rank(comm);

    /* Root sub_ranks */
    target = &(han_module->sub_ranks[(NB_TOPO_LVL-1)*root]);

    /* My sub_ranks */
    me = &(han_module->sub_ranks[(NB_TOPO_LVL-1)*w_rank]);

    /* Recursive algorithm: from higher to lower */
    for (int topo_lvl = GLOBAL_COMMUNICATOR-1 ; topo_lvl >= 0 ; topo_lvl--) {

        /* If we share all the next levels with the root, we get the data */
        int perform_bcast = 1;
        for (int i = topo_lvl-1 ; i >= 0 ; i--) {
            if (target[i] != me[i]) {
                perform_bcast = 0;
            }
        }
        if (perform_bcast) {
            sub_comm = han_module->sub_comm[topo_lvl];
            sub_comm->c_coll->coll_bcast(buff,
                                         count,
                                         dtype,
                                         target[topo_lvl],
                                         sub_comm,
                                         sub_comm->c_coll->coll_bcast_module);
        }
    }

    return OMPI_SUCCESS;
}

/* 2 levels hierarchical collective with pipeline.
 * Count is split in max segments considering MCA parameter.
 * (OMPI_MCA_coll_han_bcast_pipeline_segment_count)
 * Pipeline only starts at the size given by another MCA parameter.
 * (OMPI_MCA_coll_han_bcast_pipeline_start_size)
 * First, for each segment, a blocking INTER_NODE Bcast is performed.
 * Then, a Non-blocking INTRA_NODE Bcast is called.
 * If the split isn't perfect, we spread remaining data in all segments*/
int
mca_coll_han_bcast_intra_pipelined_2_level(void *buff,
                                           int count,
                                           struct ompi_datatype_t *dtype,
                                           int root,
                                           struct ompi_communicator_t *comm,
                                           mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    int w_rank;
    w_rank = ompi_comm_rank(comm);
    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(0, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_bcast(buff, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }
    int err;
    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_bcast(buff, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }

    /* Topo must be initialized to know rank distribution which then is used to
     * determine if han can be used */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle bcast with this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_bcast(buff, count, dtype, root,
                                        comm, comm->c_coll->coll_bcast_module);
    }
    /* Check if is it worth to use pipeline (at least 2 complete segments) */
    size_t dtype_size;
    int segment_min_count;
    int nb_segment;
    ompi_datatype_type_size(dtype, &dtype_size);
    /* We're cutting in count */
    segment_min_count = mca_coll_han_component.bcast_pipeline_start_size / dtype_size;
    nb_segment = mca_coll_han_component.bcast_pipeline_segment_count;
    if (nb_segment <= 1 || count < segment_min_count * 2 || count == 1) {
        return mca_coll_han_bcast_intra_simple(buff, count, dtype, root, comm, module);
    }
    /* Value of count for each bcast in pipeline */
    int pipeline_count;
    /* Remain count add to other segment*/ 
    int remaining_count;

    /* Case start of pipeline, try to cut with activation size of pipeline */
    int expected_nb_segment = count / segment_min_count;
    if (expected_nb_segment < nb_segment) {
        nb_segment = expected_nb_segment;
    }

    /* Take remaining data and share them in other segments */
    remaining_count = count % nb_segment;
    
    /* Cut with segment count */
    pipeline_count = count / nb_segment;

    /* sub communicator ranks information */
    const int *root_sub_ranks;
    const int *my_sub_ranks;
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    /* Root sub_ranks */
    root_sub_ranks = mca_coll_han_get_sub_ranks(han_module, root);
    /* My sub_ranks */
    my_sub_ranks = mca_coll_han_get_sub_ranks(han_module, w_rank);

    /* Pipeline */
    ompi_request_t **req;
    req = malloc(nb_segment * sizeof(struct ompi_request_t *));
    int pipeline_count_with_remain = pipeline_count + 1;
    ptrdiff_t dtype_extent;
    ompi_datatype_type_extent(dtype, &dtype_extent);
    for (int segment = 0; segment < nb_segment; segment++) {
        int bcast_count;
        int offset;
        if (segment < remaining_count) {
            bcast_count = pipeline_count_with_remain;
            offset = segment * pipeline_count_with_remain;
        } else {
            bcast_count = pipeline_count;
            offset = pipeline_count * segment + remaining_count;
        }
        if (root_sub_ranks[LEAF_LEVEL] == my_sub_ranks[LEAF_LEVEL]) {
            up_comm->c_coll->coll_bcast(buff + offset * dtype_extent,
                                        bcast_count,
                                        dtype,
                                        root_sub_ranks[INTER_NODE],
                                        up_comm,
                                        up_comm->c_coll->coll_bcast_module);
        }
        low_comm->c_coll->coll_ibcast(buff + offset * dtype_extent,
                                      bcast_count,
                                      dtype,
                                      root_sub_ranks[LEAF_LEVEL],
                                      low_comm,
                                      &req[segment],
                                      low_comm->c_coll->coll_ibcast_module);
    }
    ompi_request_wait_all(nb_segment, req, MPI_STATUS_IGNORE);
    free(req);
    return OMPI_SUCCESS;
}
