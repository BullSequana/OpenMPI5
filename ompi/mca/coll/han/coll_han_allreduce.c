/*
 * Copyright (c) 2018-2023 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 *
 * Copyright (c) 2020      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2022      IBM Corporation. All rights reserved
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * This files contains all the hierarchical implementations of allreduce
 * Only work with regular situation (each node has equal number of processes)
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

static int mca_coll_han_allreduce_t0_task(void *task_args);
static int mca_coll_han_allreduce_t1_task(void *task_args);
static int mca_coll_han_allreduce_t2_task(void *task_args);
static int mca_coll_han_allreduce_t3_task(void *task_args);

/* Only work with regular situation (each node has equal number of processes) */

static inline void
mca_coll_han_set_allreduce_args(mca_coll_han_allreduce_args_t * args,
                                mca_coll_task_t * cur_task,
                                void *sbuf,
                                void *rbuf,
                                int seg_count,
                                struct ompi_datatype_t *dtype,
                                struct ompi_op_t *op,
                                int root_up_rank,
                                int root_low_rank,
                                struct ompi_communicator_t *up_comm,
                                struct ompi_communicator_t *low_comm,
                                int num_segments,
                                int cur_seg,
                                int w_rank,
                                int last_seg_count,
                                bool noop, ompi_request_t * req, int *completed)
{
    args->cur_task = cur_task;
    args->sbuf = sbuf;
    args->rbuf = rbuf;
    args->seg_count = seg_count;
    args->dtype = dtype;
    args->op = op;
    args->root_up_rank = root_up_rank;
    args->root_low_rank = root_low_rank;
    args->up_comm = up_comm;
    args->low_comm = low_comm;
    args->num_segments = num_segments;
    args->cur_seg = cur_seg;
    args->w_rank = w_rank;
    args->last_seg_count = last_seg_count;
    args->noop = noop;
    args->req = req;
    args->completed = completed;
}

/*
 * Each segment of the message needs to go though 4 steps to perform MPI_Allreduce:
 *     lr: lower level (shared-memory or intra-node) reduce,
 *     ur: upper level (inter-node) reduce,
 *     ub: upper level (inter-node) bcast,
 *     lb: lower level (shared-memory or intra-node) bcast.
 * Hence, in each iteration, there is a combination of collective operations which is called a task.
 *        | seg 0 | seg 1 | seg 2 | seg 3 |
 * iter 0 |  lr   |       |       |       | task: t0, contains lr
 * iter 1 |  ur   |  lr   |       |       | task: t1, contains ur and lr
 * iter 2 |  ub   |  ur   |  lr   |       | task: t2, contains ub, ur and lr
 * iter 3 |  lb   |  ub   |  ur   |  lr   | task: t3, contains lb, ub, ur and lr
 * iter 4 |       |  lb   |  ub   |  ur   | task: t3, contains lb, ub and ur
 * iter 5 |       |       |  lb   |  ub   | task: t3, contains lb and ub
 * iter 6 |       |       |       |  lb   | task: t3, contains lb
 */

int
mca_coll_han_allreduce_intra(const void *sbuf,
                             void *rbuf,
                             int count,
                             struct ompi_datatype_t *dtype,
                             struct ompi_op_t *op,
                             struct ompi_communicator_t *comm, mca_coll_base_module_t * module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if(!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this communicator (not 2 levels). Drop HAN support in this communicator and fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf, rbuf, count, dtype, op,
                                            comm, comm->c_coll->coll_reduce_module);
    }

    /* No support for non-commutative operations */
    if(!ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this operation. Fall back on another component\n"));
        goto prev_allreduce_intra;
    }

    /* Create the subcommunicators */
    if( OMPI_SUCCESS != mca_coll_han_comm_create(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this communicator. Drop HAN support in this communicator and fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    ptrdiff_t extent, lb;
    size_t dtype_size;
    ompi_datatype_get_extent(dtype, &lb, &extent);
    int seg_count = count, w_rank;
    w_rank = ompi_comm_rank(comm);
    ompi_datatype_type_size(dtype, &dtype_size);

    ompi_communicator_t *low_comm;
    ompi_communicator_t *up_comm;

    /* use MCA parameters for now */
    low_comm = han_module->cached_low_comms[mca_coll_han_component.han_allreduce_low_module];
    up_comm = han_module->cached_up_comms[mca_coll_han_component.han_allreduce_up_module];
    COLL_BASE_COMPUTED_SEGCOUNT(mca_coll_han_component.han_allreduce_segsize, dtype_size,
                                seg_count);

    /* Determine number of elements sent per task. */
    OPAL_OUTPUT_VERBOSE((10, mca_coll_han_component.han_output,
                         "In HAN Allreduce seg_size %d seg_count %d count %d\n",
                         mca_coll_han_component.han_allreduce_segsize, seg_count, count));
    int num_segments = (count + seg_count - 1) / seg_count;

    int low_rank = ompi_comm_rank(low_comm);
    int root_up_rank = 0;
    int root_low_rank = 0;
    /* Create t0 task for the first segment */
    mca_coll_task_t *t0 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t0 task arguments */
    int *completed = (int *) malloc(sizeof(int));
    completed[0] = 0;
    mca_coll_han_allreduce_args_t *t = malloc(sizeof(mca_coll_han_allreduce_args_t));
    mca_coll_han_set_allreduce_args(t, t0, (char *) sbuf, (char *) rbuf, seg_count, dtype, op,
                                    root_up_rank, root_low_rank, up_comm, low_comm, num_segments, 0,
                                    w_rank, count - (num_segments - 1) * seg_count,
                                    low_rank != root_low_rank, NULL, completed);
    /* Init t0 task */
    init_task(t0, mca_coll_han_allreduce_t0_task, (void *) (t));
    /* Issure t0 task */
    issue_task(t0);

    /* Create t1 tasks for the current segment */
    mca_coll_task_t *t1 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t1 task arguments */
    t->cur_task = t1;
    /* Init t1 task */
    init_task(t1, mca_coll_han_allreduce_t1_task, (void *) t);
    /* Issue t1 task */
    issue_task(t1);

    /* Create t2 tasks for the current segment */
    mca_coll_task_t *t2 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t2 task arguments */
    t->cur_task = t2;
    /* Init t2 task */
    init_task(t2, mca_coll_han_allreduce_t2_task, (void *) t);
    issue_task(t2);

    /* Create t3 tasks for the current segment */
    mca_coll_task_t *t3 = OBJ_NEW(mca_coll_task_t);
    /* Setup up t3 task arguments */
    t->cur_task = t3;
    /* Init t3 task */
    init_task(t3, mca_coll_han_allreduce_t3_task, (void *) t);
    issue_task(t3);

    while (t->completed[0] != t->num_segments) {
        /* Create t_next_seg tasks for the current segment */
        mca_coll_task_t *t_next_seg = OBJ_NEW(mca_coll_task_t);
        /* Setup up t_next_seg task arguments */
        t->cur_task = t_next_seg;
        t->sbuf = (t->sbuf == MPI_IN_PLACE) ? MPI_IN_PLACE : (char *) t->sbuf + extent * t->seg_count;
        t->rbuf = (char *) t->rbuf + extent * t->seg_count;
        t->cur_seg = t->cur_seg + 1;
        /* Init t_next_seg task */
        init_task(t_next_seg, mca_coll_han_allreduce_t3_task, (void *) t);
        issue_task(t_next_seg);
    }
    free(t->completed);
    t->completed = NULL;
    free(t);

    return OMPI_SUCCESS;

 prev_allreduce_intra:
    return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                          comm, han_module->previous_allreduce_module);
}

/* t0 task that performs a local reduction */
int mca_coll_han_allreduce_t0_task(void *task_args)
{
    mca_coll_han_allreduce_args_t *t = (mca_coll_han_allreduce_args_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Allreduce:  t0 %d r_buf %d\n", t->w_rank, t->cur_seg,
                         ((int *) t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    if (MPI_IN_PLACE == t->sbuf) {
        if (!t->noop) {
            t->low_comm->c_coll->coll_reduce(MPI_IN_PLACE, (char *) t->rbuf, t->seg_count, t->dtype,
                                             t->op, t->root_low_rank, t->low_comm,
                                             t->low_comm->c_coll->coll_reduce_module);
        }
        else {
            t->low_comm->c_coll->coll_reduce((char *) t->rbuf, NULL, t->seg_count, t->dtype,
                                             t->op, t->root_low_rank, t->low_comm,
                                             t->low_comm->c_coll->coll_reduce_module);
        }
    }
    else {
        t->low_comm->c_coll->coll_reduce((char *) t->sbuf, (char *) t->rbuf, t->seg_count, t->dtype,
                                         t->op, t->root_low_rank, t->low_comm,
                                         t->low_comm->c_coll->coll_reduce_module);
    }
    return OMPI_SUCCESS;
}

/* t1 task that performs a ireduce on top communicator */
int mca_coll_han_allreduce_t1_task(void *task_args)
{
    mca_coll_han_allreduce_args_t *t = (mca_coll_han_allreduce_args_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Allreduce:  t1 %d r_buf %d\n", t->w_rank, t->cur_seg,
                         ((int *) t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *ireduce_req;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ur of cur_seg */
        if (up_rank == t->root_up_rank) {
            t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE, (char *) t->rbuf, t->seg_count, t->dtype,
                                             t->op, t->root_up_rank, t->up_comm, &ireduce_req,
                                             t->up_comm->c_coll->coll_ireduce_module);
        } else {
            t->up_comm->c_coll->coll_ireduce((char *) t->rbuf, (char *) t->rbuf, t->seg_count,
                                             t->dtype, t->op, t->root_up_rank, t->up_comm,
                                             &ireduce_req, t->up_comm->c_coll->coll_ireduce_module);
        }
    }
    /* lr of cur_seg+1 */
    if (t->cur_seg <= t->num_segments - 2) {
        if (t->cur_seg == t->num_segments - 2 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }

        if (t->sbuf == MPI_IN_PLACE) {
            if (!t->noop) {
                t->low_comm->c_coll->coll_reduce(MPI_IN_PLACE,
                                                 (char *) t->rbuf + extent * t->seg_count, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);
            } else {
                t->low_comm->c_coll->coll_reduce((char *) t->rbuf + extent * t->seg_count,
                                                 NULL, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);

            }
        } else {
            t->low_comm->c_coll->coll_reduce((char *) t->sbuf + extent * t->seg_count,
                                             (char *) t->rbuf + extent * t->seg_count, tmp_count,
                                             t->dtype, t->op, t->root_low_rank, t->low_comm,
                                             t->low_comm->c_coll->coll_reduce_module);
	}
    }
    if (!t->noop) {
        ompi_request_wait(&ireduce_req, MPI_STATUS_IGNORE);
    }

    return OMPI_SUCCESS;
}

/* t2 task */
int mca_coll_han_allreduce_t2_task(void *task_args)
{
    mca_coll_han_allreduce_args_t *t = (mca_coll_han_allreduce_args_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Allreduce:  t2 %d r_buf %d\n", t->w_rank, t->cur_seg,
                         ((int *) t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *reqs[2];
    int req_count = 0;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ub of cur_seg */
        t->up_comm->c_coll->coll_ibcast((char *) t->rbuf, t->seg_count, t->dtype, t->root_up_rank,
                                        t->up_comm, &(reqs[0]),
                                        t->up_comm->c_coll->coll_ibcast_module);
        req_count++;
        /* ur of cur_seg+1 */
        if (t->cur_seg <= t->num_segments - 2) {
            if (t->cur_seg == t->num_segments - 2 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            if (up_rank == t->root_up_rank) {
                t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE,
                                                 (char *) t->rbuf + extent * t->seg_count,
                                                 tmp_count, t->dtype, t->op, t->root_up_rank,
                                                 t->up_comm, &(reqs[1]),
                                                 t->up_comm->c_coll->coll_ireduce_module);
            } else {
                t->up_comm->c_coll->coll_ireduce((char *) t->rbuf + extent * t->seg_count,
                                                 (char *) t->rbuf + extent * t->seg_count,
                                                 tmp_count, t->dtype, t->op, t->root_up_rank,
                                                 t->up_comm, &(reqs[1]),
                                                 t->up_comm->c_coll->coll_ireduce_module);
            }
            req_count++;
        }
    }
    /* lr of cur_seg+2 */
    if (t->cur_seg <= t->num_segments - 3) {
        if (t->cur_seg == t->num_segments - 3 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }

	if (t->sbuf == MPI_IN_PLACE) {
	    if (!t->noop) {
                t->low_comm->c_coll->coll_reduce(MPI_IN_PLACE,
                                                 (char *) t->rbuf + 2 * extent * t->seg_count, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);
	    } else {
                t->low_comm->c_coll->coll_reduce((char *) t->rbuf + 2 * extent * t->seg_count,
                                                 NULL, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);

	    }
	} else {
            t->low_comm->c_coll->coll_reduce((char *) t->sbuf + 2 * extent * t->seg_count,
                                             (char *) t->rbuf + 2 * extent * t->seg_count, tmp_count,
                                             t->dtype, t->op, t->root_low_rank, t->low_comm,
                                             t->low_comm->c_coll->coll_reduce_module);
	}
    }
    if (!t->noop && req_count > 0) {
        ompi_request_wait_all(req_count, reqs, MPI_STATUSES_IGNORE);
    }


    return OMPI_SUCCESS;
}

/* t3 task that performs broadcasts */
int mca_coll_han_allreduce_t3_task(void *task_args)
{
    mca_coll_han_allreduce_args_t *t = (mca_coll_han_allreduce_args_t *) task_args;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Allreduce:  t3 %d r_buf %d\n", t->w_rank, t->cur_seg,
                         ((int *) t->rbuf)[0]));
    OBJ_RELEASE(t->cur_task);
    ptrdiff_t extent, lb;
    ompi_datatype_get_extent(t->dtype, &lb, &extent);
    ompi_request_t *reqs[2];
    int req_count = 0;
    int tmp_count = t->seg_count;
    if (!t->noop) {
        int up_rank = ompi_comm_rank(t->up_comm);
        /* ub of cur_seg+1 */
        if (t->cur_seg <= t->num_segments - 2) {
            if (t->cur_seg == t->num_segments - 2 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            t->up_comm->c_coll->coll_ibcast((char *) t->rbuf + extent * t->seg_count, tmp_count,
                                            t->dtype, t->root_up_rank, t->up_comm, &(reqs[0]),
                                            t->up_comm->c_coll->coll_ibcast_module);
            req_count++;
        }
        /* ur of cur_seg+2 */
        if (t->cur_seg <= t->num_segments - 3) {
            if (t->cur_seg == t->num_segments - 3 && t->last_seg_count != t->seg_count) {
                tmp_count = t->last_seg_count;
            }
            if (up_rank == t->root_up_rank) {
                t->up_comm->c_coll->coll_ireduce(MPI_IN_PLACE,
                                                 (char *) t->rbuf + 2 * extent * t->seg_count,
                                                 tmp_count, t->dtype, t->op, t->root_up_rank,
                                                 t->up_comm, &(reqs[1]),
                                                 t->up_comm->c_coll->coll_ireduce_module);
            } else {
                t->up_comm->c_coll->coll_ireduce((char *) t->rbuf + 2 * extent * t->seg_count,
                                                 (char *) t->rbuf + 2 * extent * t->seg_count,
                                                 tmp_count, t->dtype, t->op, t->root_up_rank,
                                                 t->up_comm, &(reqs[1]),
                                                 t->up_comm->c_coll->coll_ireduce_module);
            }
            req_count++;
        }
    }
    /* lr of cur_seg+3 */
    if (t->cur_seg <= t->num_segments - 4) {
        if (t->cur_seg == t->num_segments - 4 && t->last_seg_count != t->seg_count) {
            tmp_count = t->last_seg_count;
        }

        if (t->sbuf == MPI_IN_PLACE) {
            if (!t->noop) {
                t->low_comm->c_coll->coll_reduce(MPI_IN_PLACE,
                                                 (char *) t->rbuf + 3 * extent * t->seg_count, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);
	    } else {
                t->low_comm->c_coll->coll_reduce((char *) t->rbuf + 3 * extent * t->seg_count,
                                                 NULL, tmp_count,
                                                 t->dtype, t->op, t->root_low_rank, t->low_comm,
                                                 t->low_comm->c_coll->coll_reduce_module);
            }
        } else {
            t->low_comm->c_coll->coll_reduce((char *) t->sbuf + 3 * extent * t->seg_count,
                                             (char *) t->rbuf + 3 * extent * t->seg_count, tmp_count,
                                             t->dtype, t->op, t->root_low_rank, t->low_comm,
                                             t->low_comm->c_coll->coll_reduce_module);
        }
    }
    /* lb of cur_seg */
    if (t->cur_seg == t->num_segments - 1 && t->last_seg_count != t->seg_count) {
        tmp_count = t->last_seg_count;
    } else {
        tmp_count = t->seg_count;
    }

    t->low_comm->c_coll->coll_bcast((char *) t->rbuf, tmp_count, t->dtype, t->root_low_rank,
                                    t->low_comm, t->low_comm->c_coll->coll_bcast_module);
    if (!t->noop && req_count > 0) {
        ompi_request_wait_all(req_count, reqs, MPI_STATUSES_IGNORE);
    }

    t->completed[0]++;
    OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                         "[%d] HAN Allreduce:  t3 %d total %d\n", t->w_rank, t->cur_seg,
                         t->completed[0]));

    return OMPI_SUCCESS;
}

/*
 * Short implementation of allreduce that only does hierarchical
 * communications without tasks.
 */
int
mca_coll_han_allreduce_intra_simple(const void *sbuf,
                                    void *rbuf,
                                    int count,
                                    struct ompi_datatype_t *dtype,
                                    struct ompi_op_t *op,
                                    struct ompi_communicator_t *comm,
                                    mca_coll_base_module_t *module)
{
    ompi_communicator_t *low_comm;
    ompi_communicator_t *up_comm;
    int root_low_rank = 0;
    int low_rank;
    int ret;
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
#if OPAL_ENABLE_DEBUG
    mca_coll_han_component_t *cs = &mca_coll_han_component;
#endif

    OPAL_OUTPUT_VERBOSE((10, cs->han_output,
                    "[OMPI][han] in mca_coll_han_reduce_intra_simple\n"));

    if(!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this communicator (not 2 levels). Drop HAN support in this communicator and fall back on another component\n");
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf, rbuf, count, dtype, op,
                                            comm, comm->c_coll->coll_reduce_module);
    }

    // Fallback to another component if the op cannot commute
    if (! ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this operation. Fall back on another component\n"));
        goto prev_allreduce;
    }

    /* Create the subcommunicators */
    if( OMPI_SUCCESS != mca_coll_han_comm_create_new(comm, han_module) ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with this communicator. Drop HAN support in this communicator and fall back on another component\n"));
        /* HAN cannot work with this communicator so fallback on all collectives */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    low_comm = han_module->sub_comm[LEAF_LEVEL];
    up_comm = han_module->sub_comm[INTER_NODE];
    low_rank = ompi_comm_rank(low_comm);

    /* Low_comm reduce */
    if (MPI_IN_PLACE == sbuf) {
        if (low_rank == root_low_rank) {
            ret = low_comm->c_coll->coll_reduce(MPI_IN_PLACE, (char *)rbuf,
                count, dtype, op, root_low_rank,
                low_comm, low_comm->c_coll->coll_reduce_module);
        }
        else {
            ret = low_comm->c_coll->coll_reduce((char *)rbuf, NULL,
                count, dtype, op, root_low_rank,
                low_comm, low_comm->c_coll->coll_reduce_module);
        }
    }
    else {
        ret = low_comm->c_coll->coll_reduce((char *)sbuf, (char *)rbuf,
                count, dtype, op, root_low_rank,
                low_comm, low_comm->c_coll->coll_reduce_module);
    }
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        OPAL_OUTPUT_VERBOSE((30, cs->han_output,
                             "HAN/ALLREDUCE: low comm reduce failed. "
                             "Falling back to another component\n"));
        goto prev_allreduce;
    }

    /* Local roots perform a allreduce on the upper comm */
    if (low_rank == root_low_rank) {
        ret = up_comm->c_coll->coll_allreduce(MPI_IN_PLACE, rbuf, count, dtype, op,
                    up_comm, up_comm->c_coll->coll_allreduce_module);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
            OPAL_OUTPUT_VERBOSE((30, cs->han_output,
                             "HAN/ALLREDUCE: up comm allreduce failed. \n"));
            /*
             * Do not fallback in such a case: only root_low_ranks follow this
             * path, the other ranks are in another collective.
             * ==> Falling back would potentially lead to a hang.
             * Simply return the error
             */
            return ret;
        }
    }

    /* Low_comm bcast */
    ret = low_comm->c_coll->coll_bcast(rbuf, count, dtype,
                root_low_rank, low_comm, low_comm->c_coll->coll_bcast_module);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != ret)) {
        OPAL_OUTPUT_VERBOSE((30, cs->han_output,
                             "HAN/ALLREDUCE: low comm bcast failed. "
                             "Falling back to another component\n"));
        goto prev_allreduce;
    }

    return OMPI_SUCCESS;

 prev_allreduce:
    return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                          comm, han_module->previous_allreduce_module);
}

/* Find a fallback on reproducible algorithm
 * use tuned, or if impossible whatever available
 */
int
mca_coll_han_allreduce_reproducible_decision(struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    int w_rank = ompi_comm_rank(comm);
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    /* populate previous modules_storage*/
    mca_coll_han_get_all_coll_modules(comm, han_module);

    /* try availability of reproducible modules*/
    int fallbacks[] = {TUNED, BASIC};
    int fallbacks_len = sizeof(fallbacks) / sizeof(*fallbacks);
    int i;
    for (i=0; i<fallbacks_len; i++) {
        int fallback = fallbacks[i];
        mca_coll_base_module_t *fallback_module
            = han_module->modules_storage.modules[fallback].module_handler;
        if (NULL != fallback_module && NULL != fallback_module->coll_allreduce) {
            if (0 == w_rank) {
                opal_output_verbose(30, mca_coll_han_component.han_output,
                                    "coll:han:allreduce_reproducible: "
                                    "fallback on %s\n",
                                    ompi_coll_han_available_components[fallback].component_name);
            }
            han_module->reproducible_allreduce_module = fallback_module;
            han_module->reproducible_allreduce = fallback_module->coll_allreduce;
            return OMPI_SUCCESS;
        }
    }
    /* fallback of the fallback */
    if (0 == w_rank) {
        opal_output_verbose(5, mca_coll_han_component.han_output,
                            "coll:han:allreduce_reproducible_decision: "
                            "no reproducible fallback\n");
    }
    han_module->reproducible_allreduce_module = han_module->previous_allreduce_module;
    han_module->reproducible_allreduce = han_module->previous_allreduce;
    return OMPI_SUCCESS;
}

/* Fallback on reproducible algorithm */
int
mca_coll_han_allreduce_reproducible(const void *sbuf,
                                    void *rbuf,
                                     int count,
                                     struct ompi_datatype_t *dtype,
                                     struct ompi_op_t *op,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    return han_module->reproducible_allreduce(sbuf, rbuf, count, dtype,
                                              op, comm,
                                              han_module
                                              ->reproducible_allreduce_module);
}


/* Basic n-levels adaptative implementation
 * Ascending reduce + descending bcast
 *
 * Example: 12 ranks on 3 nodes
 *
 * 0     4     8
 * 1     5     9
 * 2     6    10
 * 3     7    11
 *
 * 0  1  2  3  4  5  6  7  8  9  10  11
 * |_/  /  /   |_/  /  /   |_/   /   /
 * |___/  /    |___/  /    |____/   /       Intra-node reduce
 * |_____/     |_____/     |_______/
 * |           |           |
 * 0  1  2  3  4  5  6  7  8  9  10  11
 *  \          |          /
 *   \_________|_________/
 *    _________|_________                   Inter-node allreduce
 *   /         |         \
 *  /          |          \
 * 0  1  2  3  4  5  6  7  8  9  10  11
 * |           |           |
 * |_____      |_____      |______
 * |___  \     |___  \     |____   \
 * |_  \  \    |_  \  \    |_   \   \       Intra-node bcast
 * | \  \  \   | \  \  \   | \   \   \
 * 0  1  2  3  4  5  6  7  8  9  10  11
 *
 */
int
mca_coll_han_allreduce_recursive_reduce_bcast(const void *sbuf,
                                              void *rbuf,
                                              int count,
                                              struct ompi_datatype_t *dtype,
                                              struct ompi_op_t *op,
                                              struct ompi_communicator_t *comm,
                                              mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err;

    // Fallback to another component if the op cannot commute
    if (! ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with non-commutative operations. "
                             "Fall back on another component\n"));
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle be run with this communicator."
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    /* Cannot run if ppn are imbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle allreduce with this communicator (imbalance). "
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    int topo_lvl;
    ompi_communicator_t *sub_comm;
    const int *sub_ranks = mca_coll_han_get_sub_ranks(han_module, ompi_comm_rank(comm));

    /* Climbing reduce */
    for (topo_lvl = LEAF_LEVEL ; topo_lvl < GLOBAL_COMMUNICATOR ; topo_lvl++) {
        sub_comm = han_module->sub_comm[topo_lvl];

        const void *tsbuf;
        void *trbuf;

        if (LEAF_LEVEL == topo_lvl) {
            /* Keep user args for first reduce */
            if (MPI_IN_PLACE == sbuf && 0 != sub_ranks[topo_lvl]) {
                /* Cannot use MPI_IN_PLACE as sbuf if I am not root */
                tsbuf = rbuf;
                trbuf = NULL;
            } else {
                tsbuf = sbuf;
                trbuf = rbuf;
            }
        } else {
            if (0 == sub_ranks[topo_lvl]) {
                /* MPI_IN_PLACE reduce on rbuf */
                tsbuf = MPI_IN_PLACE;
                trbuf = rbuf;
            } else {
                tsbuf = rbuf;
            }
        }

        sub_comm->c_coll->coll_reduce(tsbuf,
                                      trbuf,
                                      count,
                                      dtype,
                                      op,
                                      0,
                                      sub_comm,
                                      sub_comm->c_coll->coll_reduce_module);

        /* 0 will be our delegate for upper levels */
        if (0 != sub_ranks[topo_lvl]) {
            /* It is my last reduce, wait for bcast */
            break;
        }
    }

    if (GLOBAL_COMMUNICATOR == topo_lvl) {
        /* Only final root should enter this */
        topo_lvl--;
    }

    /* Diving bcast, I start where I stopped reduce */
    for (; topo_lvl >= LEAF_LEVEL ; topo_lvl--) {
        sub_comm = han_module->sub_comm[topo_lvl];

        sub_comm->c_coll->coll_bcast(rbuf,
                                     count,
                                     dtype,
                                     0,
                                     sub_comm,
                                     sub_comm->c_coll->coll_bcast_module);
    }

    return OMPI_SUCCESS;
}

/* Basic n-levels adaptative implementation
 * Ascending allreduce
 *
 * Example: 12 ranks on 3 nodes
 *
 * 0     4     8
 * 1     5     9
 * 2     6    10
 * 3     7    11
 *
 *
 * 0     4     8
 * |     |     |
 * 1     5     9
 * |     |     |         Intra-node allreduce
 * 2     6    10
 * |     |     |
 * 3     7    11
 *
 * 0-----4-----8
 *
 * 1-----5-----9
 *                       Intra-node allreduce
 * 2-----6----10
 *
 * 3-----7----11
 */
int
mca_coll_han_allreduce_recursive_ascending(const void *sbuf,
                                           void *rbuf,
                                           int count,
                                           struct ompi_datatype_t *dtype,
                                           struct ompi_op_t *op,
                                           struct ompi_communicator_t *comm,
                                           mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err;

    // Fallback to another component if the op cannot commute
    if (! ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with non-commutative operations. "
                             "Fall back on another component\n"));
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle be run with this communicator."
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    /* Cannot run if ppn are imbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle allreduce with this communicator (imbalance). "
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    ompi_communicator_t *sub_comm;

    /* Climbing allreduce */
    for (int topo_lvl = LEAF_LEVEL ; topo_lvl < GLOBAL_COMMUNICATOR ; topo_lvl++) {
        sub_comm = han_module->sub_comm[topo_lvl];

        sub_comm->c_coll->coll_allreduce(LEAF_LEVEL == topo_lvl ? sbuf : MPI_IN_PLACE,
                                         rbuf,
                                         count,
                                         dtype,
                                         op,
                                         sub_comm,
                                         sub_comm->c_coll->coll_allreduce_module);
    }

    return OMPI_SUCCESS;
}

/* Basic n-levels adaptative implementation
 * Descending allreduce
 *
 * Example: 12 ranks on 3 nodes
 *
 * 0     4     8
 * 1     5     9
 * 2     6    10
 * 3     7    11
 *
 *
 * 0-----4-----8
 *
 * 1-----5-----9
 *                       Intra-node allreduce
 * 2-----6----10
 *
 * 3-----7----11
 *
 * 0     4     8
 * |     |     |
 * 1     5     9
 * |     |     |         Intra-node allreduce
 * 2     6    10
 * |     |     |
 * 3     7    11
 */
int
mca_coll_han_allreduce_recursive_descending(const void *sbuf,
                                            void *rbuf,
                                            int count,
                                            struct ompi_datatype_t *dtype,
                                            struct ompi_op_t *op,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err;

    // Fallback to another component if the op cannot commute
    if (! ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with non-commutative operations. "
                             "Fall back on another component\n"));
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle be run with this communicator."
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    /* Cannot run if ppn are imbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle allreduce with this communicator (imbalance). "
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    ompi_communicator_t *sub_comm;

    /* Diving allreduce */
    for (int topo_lvl = GLOBAL_COMMUNICATOR-1 ; topo_lvl >= LEAF_LEVEL ; topo_lvl--) {
        sub_comm = han_module->sub_comm[topo_lvl];

        sub_comm->c_coll->coll_allreduce(GLOBAL_COMMUNICATOR-1 == topo_lvl ? sbuf : MPI_IN_PLACE,
                                         rbuf,
                                         count,
                                         dtype,
                                         op,
                                         sub_comm,
                                         sub_comm->c_coll->coll_allreduce_module);
    }

    return OMPI_SUCCESS;
}

/* Complex n-levels adaptative implementation
 * Split allreduce into reduce_scatter + allgatherv
 *
 * Example: 12 ranks on 4 nodes on 2 clusters, count = 36
 *
 * 0  3      6  9
 * 1  4      7 10
 * 2  5      8 11
 *
 *
 * We will follow rank 4 communications
 * Step 1: Climbing reduce_scatter
 *
 * Intra-node reduce_scatter
 * Intra-node communicator (leaf_level): 3 4 5
 * count = 36, cut message through reduce_scatter in 3 parts of 12 datatypes
 *
 *         3                 4                 5
 * | 12 | 12 | 12 |  | 12 | 12 | 12 |  | 12 | 12 | 12 |
 *    |    \    \      /     |    \      /    /     |
 *    | ____\____\____/______|_____\____/    /      |
 *    |/     \    \          |      \       /       |
 *    |       \____\________ | ______\_____/        |
 *    |             \       \|/       \             |
 *    |              \_______|_________\___________ |
 *    |                      |                     \|
 *    |                      |                      |
 * | 12 | xx | xx |  | xx | 12 | xx |  | xx | xx | 12 |
 * 
 * For now, I am only in charge of the inner part of the buffer
 *
 * Inter-node reduce_scatter
 * Inter-node communicator (inter_node): 1 4
 * count = 12, cut message through reduce_scatter in 2 parts of 6 datatypes
 *     1          4
 * | 6 | 6 |  | 6 | 6 |
 *   |   \     /    |
 *   | ___\___/     |
 *   |/    \        |
 *   |      \______ |
 *   |             \|
 *   |              |
 * | 6 | x |  | x | 6 |
 *
 * For now, I am only in charge of the end of the buffer
 *
 * Step 2: upper level allreduce
 * Inter-cluster communicator (gateway): 4 10
 * This the upper level, use allreduce
 *   4     10
 * | 6 |  | 6 |
 *    \    /
 *     \  /
 *      \/
 *      /\
 *     /  \
 *    /    \
 * | 6 |  | 6 |
 *
 * Now, my contribution is computed
 *
 * Step 3: Dive back to retieve other contiburions through allgatherv
 * Inter-node communicator (inter_node): 1 4
 *     1          4
 * | 6 | x |  | x | 6 |
 *   |              |
 *   |       ______/|
 *   |      /       |
 *   |\____/__      |
 *   |    /   \     |
 *   |   /     \    |
 * | 6 | 6 |  | 6 | 6 |
 *
 * Intra-node communicator (leaf_level): 3 4 5
 *
 *         3                 4                 5
 * | 12 | xx | xx |  | xx | 12 | xx |  | xx | xx | 12 |
 *    |                      |                      |
 *    |               _______|_____________________/|
 *    |              /       |         /            |
 *    |        _____/_______/|\_______/____         |
 *    |       /    /         |       /     \        |
 *    |\_____/___ /__________|______/___    \       |
 *    |     /    /    \      |     /    \    \      |
 *    |    /    /      \     |    /      \    \     |
 * | 12 | 12 | 12 |  | 12 | 12 | 12 |  | 12 | 12 | 12 |
 *
 */
int
mca_coll_han_allreduce_recursive_scattering(const void *sbuf,
                                            void *rbuf,
                                            int count,
                                            struct ompi_datatype_t *dtype,
                                            struct ompi_op_t *op,
                                            struct ompi_communicator_t *comm,
                                            mca_coll_base_module_t *module)
{
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int err;

    // Fallback to another component if the op cannot commute
    if (! ompi_op_is_commute(op)) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle allreduce with non-commutative operations. "
                             "Fall back on another component\n"));
        return han_module->previous_allreduce(sbuf, rbuf, count, dtype, op,
                                              comm, han_module->previous_allreduce_module);
    }

    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle be run with this communicator."
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    /* Cannot run if ppn are imbalanced */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle allreduce with this communicator (imbalance). "
                            "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, allreduce);
        return comm->c_coll->coll_allreduce(sbuf,
                                            rbuf,
                                            count,
                                            dtype,
                                            op,
                                            comm,
                                            comm->c_coll->coll_allreduce_module);
    }

    /* Structure storing messages information at each level */
    struct {
        /* Start offset of our part */
        int start_off;

        /* Size of our part */
        int part_size;

        /* Count/Displacement used for both scatter of reduce_scatter and allgatherv */
        int *rcounts;
        int *displs;
    } split_infos[NB_TOPO_LVL-1];

    /* Some aliases */
    struct ompi_communicator_t **sub_comm = han_module->sub_comm;
    const int *sub_ranks = mca_coll_han_get_sub_ranks(han_module, ompi_comm_rank(comm));

    /* Extract datatype extent */
    ptrdiff_t ddt_ext;
    ompi_datatype_type_extent(dtype, &ddt_ext);

    /* Find the upper topological level where sub_comm != MPI_COMM_SELF */
    int topo_lvl;
    int upper_lvl = 0;
    for (topo_lvl = 0 ; topo_lvl < GLOBAL_COMMUNICATOR ; topo_lvl++) {
        if (ompi_comm_size(sub_comm[topo_lvl]) > 1) {
            upper_lvl = topo_lvl;
        }
    }

    /* Climbing reduce_scatter */
    for (topo_lvl = LEAF_LEVEL ; topo_lvl < upper_lvl ; topo_lvl++) {
        if(0 == topo_lvl) {
            split_infos[topo_lvl].start_off = 0;
            split_infos[topo_lvl].part_size = count;
        } else {
            split_infos[topo_lvl].start_off = split_infos[topo_lvl-1].start_off
                                              + split_infos[topo_lvl-1].displs[sub_ranks[topo_lvl-1]];
            split_infos[topo_lvl].part_size = split_infos[topo_lvl-1].rcounts[sub_ranks[topo_lvl-1]];
        }

        if (0 == split_infos[topo_lvl].part_size) {
            break;
        }

        int sub_comm_size = ompi_comm_size(sub_comm[topo_lvl]);

        split_infos[topo_lvl].rcounts = malloc(sub_comm_size * sizeof(int));
        split_infos[topo_lvl].displs = malloc(sub_comm_size * sizeof(int));

        int part = split_infos[topo_lvl].part_size / sub_comm_size;

        if (part < mca_coll_han_component.allreduce_min_recursive_split_size) {
            part = mca_coll_han_component.allreduce_min_recursive_split_size;
            int rest = split_infos[topo_lvl].part_size % part;

            int rank;
            for (rank = 0 ; rank < split_infos[topo_lvl].part_size / part ; rank++) {
                split_infos[topo_lvl].displs[rank] = rank*part;
                split_infos[topo_lvl].rcounts[rank] = part;
            }

            split_infos[topo_lvl].displs[rank] = rank*part;
            split_infos[topo_lvl].rcounts[rank] = rest;
            rank++;

            for (; rank < sub_comm_size ; rank++) {
                split_infos[topo_lvl].displs[rank] = split_infos[topo_lvl].part_size;
                split_infos[topo_lvl].rcounts[rank] = 0;
            }
        } else {
            int rest = split_infos[topo_lvl].part_size % sub_comm_size;
            for (int rank = 0 ; rank < sub_comm_size ; rank++) {
                /* Store the start of each buffer (for allgatherv) */
                split_infos[topo_lvl].displs[rank] = rank * part;
                split_infos[topo_lvl].rcounts[rank] = part;

                if (rank < rest) {
                    split_infos[topo_lvl].displs[rank] += rank;
                    split_infos[topo_lvl].rcounts[rank]++;
                } else {
                    split_infos[topo_lvl].displs[rank] += rest;
                }
            }
        }

        const char *tsbuf;
        if (0 == topo_lvl) {
            tsbuf = (const char*) sbuf;
        } else {
            tsbuf = MPI_IN_PLACE;
        }

        sub_comm[topo_lvl]->c_coll->coll_reduce_scatter(tsbuf,
                                                        rbuf,
                                                        split_infos[topo_lvl].rcounts,
                                                        dtype,
                                                        op,
                                                        sub_comm[topo_lvl],
                                                        sub_comm[topo_lvl]->c_coll->coll_reduce_scatter_module);
    }

    /* Use allreduce on upper level
     * Allows optimizations on allreduce collective
     * Avoid unwanted buffer copy on some complex cases
     */
    if (topo_lvl == upper_lvl) {
        const char *tsbuf;
        char *trbuf;
        int tcount = split_infos[topo_lvl-1].rcounts[sub_ranks[topo_lvl-1]];

        if (tcount > 0) {
            if (0 == split_infos[topo_lvl-1].start_off) {
                tsbuf = MPI_IN_PLACE;
                trbuf = rbuf;
            } else {
                tsbuf = rbuf;
                trbuf = ((char*) rbuf)
                    + ddt_ext * (split_infos[topo_lvl-1].start_off
                            + split_infos[topo_lvl-1].displs[sub_ranks[topo_lvl-1]]);
            }

            sub_comm[topo_lvl]->c_coll->coll_allreduce(tsbuf,
                                                       trbuf,
                                                       tcount,
                                                       dtype,
                                                       op,
                                                       sub_comm[topo_lvl],
                                                       sub_comm[topo_lvl]->c_coll->coll_allreduce_module);
        }
    }

    /* Diving Allgatherv */
    for (topo_lvl-- ; topo_lvl >= LEAF_LEVEL ; topo_lvl--) {
        char *trbuf;
        trbuf = ((char*) rbuf)
                + ddt_ext * split_infos[topo_lvl].start_off;

        sub_comm[topo_lvl]->c_coll->coll_allgatherv(MPI_IN_PLACE,
                                                    split_infos[topo_lvl].rcounts[sub_ranks[topo_lvl]],
                                                    dtype,
                                                    trbuf,
                                                    split_infos[topo_lvl].rcounts,
                                                    split_infos[topo_lvl].displs,
                                                    dtype,
                                                    sub_comm[topo_lvl],
                                                    sub_comm[topo_lvl]->c_coll->coll_allgatherv_module);

        free(split_infos[topo_lvl].displs);
        free(split_infos[topo_lvl].rcounts);
    }

    return OMPI_SUCCESS;
}

