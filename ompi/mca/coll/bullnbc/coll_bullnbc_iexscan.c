/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
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
#include "opal/align.h"
#include "ompi/op/op.h"

#include "coll_bullnbc_internal.h"

static inline int exscan_sched_linear(
    int rank, int comm_size, const void *sendbuf, void *recvbuf, int count,
    MPI_Datatype datatype,  MPI_Op op, char inplace, BULLNBC_Schedule *schedule,
    void *tmpbuf);
static inline int exscan_sched_recursivedoubling(
    int rank, int comm_size, const void *sendbuf, void *recvbuf,
    int count, MPI_Datatype datatype,  MPI_Op op, char inplace,
    BULLNBC_Schedule *schedule, void *tmpbuf1, void *tmpbuf2);

static mca_base_var_enum_value_t exscan_algorithms[] = {
    {0, "ignore"},
    {1, "linear"},
    {2, "recursive_doubling"},
    {0, NULL}
};

/* The following are used by dynamic and forced rules */

/* this routine is called by the component only */
/* module does not call this it calls the forced_getvalues routine instead */

int ompi_coll_bullnbc_exscan_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices)
{
  mca_base_var_enum_t *new_enum;
  int cnt;

  for( cnt = 0; NULL != exscan_algorithms[cnt].string; cnt++ );
  mca_param_indices->algorithm_count = cnt;

  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iexscan_algorithm_count",
                                         "Number of exscan algorithms available",
                                         MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                         MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_CONSTANT,
                                         &mca_param_indices->algorithm_count);

  mca_param_indices->algorithm = 0;
  (void) mca_base_var_enum_create("coll_bullnbc_exscan_algorithms", exscan_algorithms, &new_enum);
  (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                         "iexscan_algorithm",
                                         "Which exscan algorithm is used.",
                                         MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                         OPAL_INFO_LVL_5,
                                         MCA_BASE_VAR_SCOPE_ALL,
                                         &mca_param_indices->algorithm);
  mca_param_indices->segsize = 0;
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "iexscan_algorithm_segmentsize",
                                  "Segment size in bytes used by default for iexscan algorithms. Only has meaning if algorithm is forced and supports segmenting. 0 bytes means no segmentation.",
                                  MCA_BASE_VAR_TYPE_INT, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_param_indices->segsize);
  OBJ_RELEASE(new_enum);
  return OMPI_SUCCESS;
}

static int nbc_exscan_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                           struct ompi_communicator_t *comm, ompi_request_t ** request,
                           struct mca_coll_base_module_2_4_0_t *module, bool persistent) {
    int rank, p, res;
    size_t size;
    BULLNBC_Schedule *schedule;
    char inplace;
    void *tmpbuf = NULL, *tmpbuf1 = NULL, *tmpbuf2 = NULL;
    enum { NBC_EXSCAN_LINEAR, NBC_EXSCAN_RDBL } alg;
    ompi_coll_bullnbc_module_t *bullnbc_module = (ompi_coll_bullnbc_module_t*) module;
    ptrdiff_t span, gap;
    alg = NBC_EXSCAN_LINEAR;

    NBC_IN_PLACE(sendbuf, recvbuf, inplace);

    rank = ompi_comm_rank (comm);
    p = ompi_comm_size (comm);

    if (p < 2) {
        return bullnbc_get_noop_request(persistent, request);
    }

    res = ompi_datatype_type_size (datatype, &size);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }
    size *= count;
    if(mca_coll_bullnbc_component.use_dynamic_rules) {
        if(0 != mca_coll_bullnbc_component.forced_params[EXSCAN].algorithm) {
            /* if op is not commutative or MPI_IN_PLACE was specified we have to deal with it */
            alg = mca_coll_bullnbc_component.forced_params[EXSCAN].algorithm - 1; /* -1 is to shift from algorithm ID to enum */
            goto selected_rule;
        }
        if(bullnbc_module->com_rules[EXSCAN]) {
            int algorithm,dummy1,dummy2,dummy3;
            algorithm = ompi_coll_base_get_target_method_params (bullnbc_module->com_rules[EXSCAN],
                                                                 size, &dummy1, &dummy2, &dummy3);
            if(algorithm) {
                alg = algorithm - 1; /* -1 is to shift from algorithm ID to enum */
                goto selected_rule;
            }
        }
    }

selected_rule:

    span = opal_datatype_span(&datatype->super, count, &gap);
    if (alg == NBC_EXSCAN_RDBL) {
        ptrdiff_t span_align = OPAL_ALIGN(span, datatype->super.align, ptrdiff_t);
        tmpbuf = malloc(span_align + span);
        if (NULL == tmpbuf) { return OMPI_ERR_OUT_OF_RESOURCE; }
        tmpbuf1 = (void *)(-gap);
        tmpbuf2 = (char *)(span_align) - gap;
    } else {
        if (rank > 0) {
            tmpbuf = malloc(span);
            if (NULL == tmpbuf) { return OMPI_ERR_OUT_OF_RESOURCE; }
        }
    }

    schedule = OBJ_NEW(BULLNBC_Schedule);
    if (OPAL_UNLIKELY(NULL == schedule)) {
        free(tmpbuf);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (alg == NBC_EXSCAN_LINEAR) {
        res = exscan_sched_linear(rank, p, sendbuf, recvbuf, count, datatype,
                                  op, inplace, schedule, tmpbuf);
    } else {
        res = exscan_sched_recursivedoubling(rank, p, sendbuf, recvbuf, count,
                                             datatype, op, inplace, schedule, tmpbuf1, tmpbuf2);
    }
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


/*
 * exscan_sched_linear:
 *
 * Function:  Linear algorithm for exclusive scan.
 * Accepts:   Same as MPI_Iexscan
 * Returns:   MPI_SUCCESS or error code
 *
 * Working principle:
 * 1. Each process (but process 0) receives from left neighbor
 * 2. Performs op
 * 3. All but rank p - 1 do sends to it's right neighbor and exits
 *
 * Schedule length: O(1)
 */
static inline int exscan_sched_linear(
    int rank, int comm_size, const void *sendbuf, void *recvbuf, int count,
    MPI_Datatype datatype,  MPI_Op op, char inplace, BULLNBC_Schedule *schedule,
    void *tmpbuf)
{
    int res = OMPI_SUCCESS;
    ptrdiff_t gap;
    opal_datatype_span(&datatype->super, count, &gap);

    if (rank > 0) {
        if (inplace) {
            res = NBC_Sched_copy(recvbuf, false, count, datatype,
                                 (char *)tmpbuf - gap, false, count, datatype, schedule, false);
        } else {
            res = NBC_Sched_copy((void *)sendbuf, false, count, datatype,
                                 (char *)tmpbuf - gap, false, count, datatype, schedule, false);
        }
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

        res = NBC_Sched_recv(recvbuf, false, count, datatype, rank - 1, schedule, false);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

        if (rank < comm_size - 1) {
            /* We have to wait until we have the data */
            res = NBC_Sched_barrier(schedule);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            res = NBC_Sched_op(recvbuf, false, (void *)(-gap), true, count,
                               datatype, op, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            /* Send reduced data onward */
            res = NBC_Sched_send ((void *)(-gap), true, count, datatype, rank + 1, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
        }
    } else if (comm_size > 1) {
        /* Process 0 */
        if (inplace) {
            res = NBC_Sched_send(recvbuf, false, count, datatype, 1, schedule, false);
        } else {
            res = NBC_Sched_send(sendbuf, false, count, datatype, 1, schedule, false);
        }
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
    }

cleanup_and_return:
    return res;
}

/*
 * exscan_sched_recursivedoubling:
 *
 * Function:  Recursive doubling algorithm for exclusive scan.
 * Accepts:   Same as MPI_Iexscan
 * Returns:   MPI_SUCCESS or error code
 *
 * Description:  Implements recursive doubling algorithm for MPI_Iexscan.
 *               The algorithm preserves order of operations so it can
 *               be used both by commutative and non-commutative operations.
 *
 * Example for 5 processes and commutative operation MPI_SUM:
 * Process:  0                 1             2              3               4
 * recvbuf:  -                 -             -              -               -
 *   psend: [0]               [1]           [2]            [3]             [4]
 *
 *  Step 1:
 * recvbuf:  -                [0]            -             [2]              -
 *   psend: [1+0]             [0+1]         [3+2]          [2+3]           [4]
 *
 *  Step 2:
 * recvbuf:  -                [0]            [1+0]          [(0+1)+2]       -
 *   psend: [(3+2)+(1+0)]     [(2+3)+(0+1)]  [(1+0)+(3+2)]  [(1+0)+(2+3)]  [4]
 *
 *  Step 3:
 * recvbuf:  -                [0]            [1+0]          [(0+1)+2]      [(3+2)+(1+0)]
 *   psend: [4+((3+2)+(1+0))]                                              [((3+2)+(1+0))+4]
 *
 * Time complexity (worst case): \ceil(\log_2(p))(2\alpha + 2m\beta + 2m\gamma)
 * Memory requirements (per process): 2 * count * typesize = O(count)
 * Limitations: intra-communicators only
 * Schedule length: O(log(p))
 */
static inline int exscan_sched_recursivedoubling(
    int rank, int comm_size, const void *sendbuf, void *recvbuf, int count,
    MPI_Datatype datatype, MPI_Op op, char inplace,
    BULLNBC_Schedule *schedule, void *tmpbuf1, void *tmpbuf2)
{
    int res = OMPI_SUCCESS;
    char *psend = (char *)tmpbuf1;
    char *precv = (char *)tmpbuf2;

    if (!inplace) {
        res = NBC_Sched_copy((char *)sendbuf, false, count, datatype,
                             psend, true, count, datatype, schedule, true);
    } else {
        res = NBC_Sched_copy((char *)recvbuf, false, count, datatype,
                             psend, true, count, datatype, schedule, true);
    }
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

    int is_commute = ompi_op_is_commute(op);
    int is_first_block = 1;

    for (int mask = 1; mask < comm_size; mask <<= 1) {
        int remote = rank ^ mask;
        if (remote < comm_size) {
            res = NBC_Sched_send(psend, true, count, datatype, remote, schedule, false);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            res = NBC_Sched_recv(precv, true, count, datatype, remote, schedule, true);
            if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }

            if (rank > remote) {
                /* Assertion: rank > 0 and rbuf is valid */
                if (is_first_block) {
                    res = NBC_Sched_copy(precv, true, count, datatype,
                                         recvbuf, false, count, datatype, schedule, false);
                    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                    is_first_block = 0;
                } else {
                    /* Accumulate prefix reduction: recvbuf = precv <op> recvbuf */
                    res = NBC_Sched_op(precv, true, recvbuf, false, count,
                                       datatype, op, schedule, false);
                    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                }
                /* Partial result: psend = precv <op> psend */
                res = NBC_Sched_op(precv, true, psend, true, count,
                                   datatype, op, schedule, true);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
            } else {
                if (is_commute) {
                    /* psend = precv <op> psend */
                    res = NBC_Sched_op(precv, true, psend, true, count,
                                       datatype, op, schedule, true);
                    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                } else {
                    /* precv = psend <op> precv */
                    res = NBC_Sched_op(psend, true, precv, true, count,
                                       datatype, op, schedule, true);
                    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) { goto cleanup_and_return; }
                    char *tmp = psend;
                    psend = precv;
                    precv = tmp;
                }
            }
        }
    }

cleanup_and_return:
    return res;
}

int ompi_coll_bullnbc_iexscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_exscan_init(sendbuf, recvbuf, count, datatype, op,
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

int ompi_coll_bullnbc_exscan_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                 struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module) {
    int res = nbc_exscan_init(sendbuf, recvbuf, count, datatype, op,
                              comm, request, module, true);
    if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
        return res;
    }

    return OMPI_SUCCESS;
}
