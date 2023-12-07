/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpc_lowcomm.h"
#include "opal/mca/event/event.h"
#include "pml_mpc.h"
#include "pml_mpc_request.h"

static int
mca_pml_mpc_component_register(void);
static int
mca_pml_mpc_component_open(void);
static int
mca_pml_mpc_component_close(void);

static mca_pml_base_module_t*
mca_pml_mpc_component_init(int* priority,
                           bool enable_progress_threads,
                           bool enable_mpi_threads);

static int
mca_pml_mpc_component_fini(void);

mca_pml_mpc_component_t mca_pml_mpc_component = {

    /* First, fill in the super */
    .super = {
        /* First, the mca_base_component_t struct containing meta
         * information about the component itself */

        .pmlm_version = {
            MCA_PML_BASE_VERSION_2_0_0,

            .mca_component_name            = "mpc",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),
            .mca_open_component            = mca_pml_mpc_component_open,
            .mca_close_component           = mca_pml_mpc_component_close,
            .mca_register_component_params = mca_pml_mpc_component_register,
        },
        .pmlm_data = {
            /* The component is checkpoint ready */
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },

        .pmlm_init     = mca_pml_mpc_component_init,
        .pmlm_finalize = mca_pml_mpc_component_fini,
    }
};

static int
mca_pml_mpc_component_register(void)
{
    mca_pml_mpc_component.priority = 0;
    (void)mca_base_component_var_register(
                &mca_pml_mpc_component.super.pmlm_version, "priority",
                "Priority of the pml mpc component", MCA_BASE_VAR_TYPE_INT,
                NULL, 0, 0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_READONLY,
                &mca_pml_mpc_component.priority);

    mca_pml_mpc_component.verbose = 0;
    (void)mca_base_component_var_register(
                &mca_pml_mpc_component.super.pmlm_version, "verbose",
                "verbosity of the pml mpc component", MCA_BASE_VAR_TYPE_INT,
                NULL, 0, 0, OPAL_INFO_LVL_9, MCA_BASE_VAR_SCOPE_READONLY,
                &mca_pml_mpc_component.verbose);

    return OMPI_SUCCESS;
}

static int
mca_pml_mpc_component_open(void)
{
    mca_pml_mpc_component.output = opal_output_open(NULL);
    opal_output_set_verbosity(mca_pml_mpc_component.output, mca_pml_mpc_component.verbose);
    return OMPI_SUCCESS;
}

static int
mca_pml_mpc_component_close(void)
{
    opal_output_close(mca_pml_mpc_component.output);
    return OMPI_SUCCESS;
}

static mca_pml_base_module_t*
mca_pml_mpc_component_init(int* priority,
                           bool enable_progress_threads,
                           bool enable_mpi_threads)
{
    // ignore those for the moment
    (void)enable_progress_threads;
    (void)enable_mpi_threads;

    *priority = mca_pml_mpc_component.priority;
    mca_pml_mpc_component.rank = -1;

    pml_mpc_requests_free_list_init();
    pml_mpc_pending_list_init();

    mpc_lowcomm_init();
    mca_pml_mpc_component.rank = mpc_lowcomm_get_rank();

    int mpc_worldsize = mpc_lowcomm_get_size();
    pml_mpc_verbose(1, "init %d/%d priority=%d", mca_pml_mpc_component.rank,
                    mpc_worldsize, *priority);

    /* context update ? */
    return &mca_pml_mpc_module.super;
}

static int
mca_pml_mpc_component_fini(void)
{
    int mpc_worldsize = mpc_lowcomm_get_size();
    pml_mpc_verbose(5, "fini %d/%d", mca_pml_mpc_component.rank,
                    mpc_worldsize);

    mpc_lowcomm_release();
    pml_mpc_pending_list_fini();
    pml_mpc_requests_free_list_fini();
    return OMPI_SUCCESS;
}
