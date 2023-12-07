/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2023-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "opal/class/opal_list.h"

#include "ompi/mca/pml/pml.h"
#include "opal/mca/base/mca_base_var.h"
#include "opal/mca/base/mca_base_pvar.h"
#include "opal/runtime/opal_params.h"
#include "opal/util/bit_ops.h"

#include "opal/util/proc.h"

#include "pml_ob1_component.h"
#include "pml_ob1_stats.h"

/* Simple set of functions to expose performance counters throw MPI_T interface */

struct pml_ob1_stats_datastruct {
    bool use_power_of_two;
    unsigned depth_length;
    unsigned *depths;
};

static struct pml_ob1_stats_datastruct __pml_ob1_stats_data;

static void pml_ob1_stats_update_depth(const int depth)
{
    int dim, size;
    int index;

    if (__pml_ob1_stats_data.use_power_of_two) {
#if OPAL_C_HAVE_BUILTIN_CLZ
        size = 8 * sizeof(int);
        index = size - __builtin_clz(depth);
#else
        /* Not tested !! */
        for (index = 0, size = 1; size <= value; ++index, size <<= 1) /* empty */
            ;
#endif
    } else {
        index = depth - 1;
    }

    index--;

    if (index >= __pml_ob1_stats_data.depth_length) {
        index = __pml_ob1_stats_data.depth_length - 1;
    }

    __pml_ob1_stats_data.depths[index] += 1;
}

static int mca_pml_ob1_count_depth(mca_pml_ob1_comm_t *comm, mca_pml_ob1_comm_proc_t *proc,
                                   const mca_pml_sequence_t max_seq)
{
    int depth = 0;
    opal_list_t *queue;
    mca_pml_ob1_recv_request_t *request;

    /* Find expected queue */
    queue = (NULL == proc) ? &comm->wild_receives : &proc->specific_receives;

    if (NULL == queue || opal_list_get_size(queue) == 0) {
        return 0;
    }

    request = (mca_pml_ob1_recv_request_t *) opal_list_get_first(queue);

    while (NULL != request) {
        opal_list_item_t *item;
        mca_pml_sequence_t seq = request->req_recv.req_base.req_sequence;

        if (seq > max_seq) {
            break;
        }

        depth++;

        item = opal_list_get_next((opal_list_item_t *) request);
        if (opal_list_get_end(queue) == item) {
            break;
        }

        request = (mca_pml_ob1_recv_request_t *) item;
    }

    return depth;
}

void mca_pml_ob1_compute_depth(mca_pml_ob1_comm_t *comm, mca_pml_ob1_comm_proc_t *cur_proc,
                               mca_pml_ob1_recv_request_t *match)
{
    int depth = 1;
    mca_pml_sequence_t match_seq;
    const int comm_size = comm->num_procs;
    mca_pml_ob1_recv_request_t *specific_recv, *wild_recv;

    if (NULL == match) {
        match_seq = ~((mca_pml_sequence_t) 0);
    } else {
        match_seq = match->req_recv.req_base.req_sequence;
    }

    /* proc any_source recv */
    depth += mca_pml_ob1_count_depth(comm, NULL, match_seq);

    if (mca_pml_ob1.mode_portals) {
        for (int i = 0; i < comm->num_procs; i++) {
            mca_pml_ob1_comm_proc_t *proc;

            /* Get ob1_proc */
            if (NULL == (proc = comm->procs[i])) {
                continue;
            }

            depth += mca_pml_ob1_count_depth(comm, proc, match_seq);
        }
    } else {
        depth += mca_pml_ob1_count_depth(comm, cur_proc, match_seq);
    }

    pml_ob1_stats_update_depth(depth);
}

static int mca_pml_ob1_hist_notify(mca_base_pvar_t *pvar, mca_base_pvar_event_t event,
                                   void *obj_handle, int *count)
{
    switch (event) {
    case MCA_BASE_PVAR_HANDLE_BIND:
        *count = __pml_ob1_stats_data.depth_length;
        break;
    }

    return OMPI_SUCCESS;
}

static int mca_pml_ob1_matching_depth(const struct mca_base_pvar_t *pvar, void *value,
                                      void *obj_handle)
{
    unsigned *values = (unsigned *) value;
    memcpy(values, __pml_ob1_stats_data.depths,
           __pml_ob1_stats_data.depth_length * sizeof(unsigned));
    return OMPI_SUCCESS;
}

void pml_ob1_stats_register_parameters(void)
{
    __pml_ob1_stats_data.depth_length = 16;
    (void) mca_base_component_var_register(&mca_pml_ob1_component.pmlm_version,
                                           "matching_depth_length",
                                           "number of matching depth buckets",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0, OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &__pml_ob1_stats_data.depth_length);

    __pml_ob1_stats_data.depths = malloc(sizeof(unsigned) * __pml_ob1_stats_data.depth_length);
    memset(__pml_ob1_stats_data.depths, 0, sizeof(unsigned) * __pml_ob1_stats_data.depth_length);

    __pml_ob1_stats_data.use_power_of_two = true;
    (void) mca_base_component_var_register(&mca_pml_ob1_component.pmlm_version,
                                           "matching_use_power_of_two",
                                           "use buckets based on power of two",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0, OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &__pml_ob1_stats_data.use_power_of_two);

    (void) mca_base_component_pvar_register(&mca_pml_ob1_component.pmlm_version,
                                            "matching_depth_lengths", "matching depth counters",
                                            OPAL_INFO_LVL_4, MPI_T_PVAR_CLASS_SIZE,
                                            MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL,
                                            MPI_T_BIND_NO_OBJECT,
                                            MCA_BASE_PVAR_FLAG_READONLY
                                                | MCA_BASE_PVAR_FLAG_CONTINUOUS,
                                            mca_pml_ob1_matching_depth, NULL,
                                            mca_pml_ob1_hist_notify,
                                            (void *) __pml_ob1_stats_data.depths);
}
