/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2023      BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_tuned.h"
#include "ompi/mca/coll/base/coll_base_topo.h"
#include "ompi/mca/coll/base/coll_base_util.h"

static int coll_tuned_neighbor_alltoallv_forced_algorithm = 0;

/* Neighbor Alltoallv controlled algorithm limit of send mesage rate,
 * default is "blocking" send mode */
static int coll_tuned_neighbor_alltoallv_controlled_limit = 1;

/* valid values for coll_tuned_neighbor_alltoallv_forced_algorithm */
static const mca_base_var_enum_value_t neighbor_alltoallv_algorithms[] = {
    {0, "ignore"},
    {1, "basic"},
    {2, "controlled"},
    {0, NULL}
};

/*
 * The following are used by dynamic and forced rules.  Publish
 * details of each algorithm and if its forced/fixed/locked in as you add
 * methods/algorithms you must update this and the query/map routines.
 * This routine is called by the component only.  This makes sure that
 * the mca parameters are set to their initial values and perms.
 * Module does not call this.  They call the forced_getvalues routine
 * instead.
 */
int ompi_coll_tuned_neighbor_alltoallv_intra_check_forced_init(coll_tuned_force_algorithm_mca_param_indices_t
                                                               *mca_param_indices)
{
    mca_base_var_enum_t *new_enum;
    int cnt;
    for( cnt = 0; NULL != neighbor_alltoallv_algorithms[cnt].string; cnt++ );
    ompi_coll_tuned_forced_max_algorithms[NEIGHBOR_ALLTOALLV] = cnt;

    (void) mca_base_component_var_register(&mca_coll_tuned_component.super.collm_version,
                                           "neighbor_alltoallv_algorithm_count",
                                           "Number of neighbor_alltoallv algorithms available",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEFAULT_ONLY,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_CONSTANT,
                                           &ompi_coll_tuned_forced_max_algorithms[NEIGHBOR_ALLTOALLV]);

    /* MPI_T: This variable should eventually be bound to a communicator */
    coll_tuned_neighbor_alltoallv_forced_algorithm = 0;
    (void) mca_base_var_enum_create("coll_tuned_neighbor_alltoallv_algorithms", neighbor_alltoallv_algorithms, &new_enum);
    mca_param_indices->algorithm_param_index =
        mca_base_component_var_register(&mca_coll_tuned_component.super.collm_version,
                                        "neighbor_alltoallv_algorithm",
                                        "Which neighbor_alltoallv algorithm is used. "
                                        "Only relevant if coll_tuned_use_dynamic_rules is true.",
                                        MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                        OPAL_INFO_LVL_5,
                                        MCA_BASE_VAR_SCOPE_ALL,
                                        &coll_tuned_neighbor_alltoallv_forced_algorithm);
    OBJ_RELEASE(new_enum);
    if (mca_param_indices->algorithm_param_index < 0) {
        return mca_param_indices->algorithm_param_index;
    }

    coll_tuned_neighbor_alltoallv_controlled_limit = 1;
    (void) mca_base_component_var_register(&mca_coll_tuned_component.super.collm_version,
                                           "neighbor_alltoallv_controlled_limit",
                                           "Number of simultaneous send autorized for"
                                           "neighbor alltoallv algorithm."
                                           "Default is 1",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_SETTABLE,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_ALL,
                                           &coll_tuned_neighbor_alltoallv_controlled_limit);
    return (MPI_SUCCESS);
}

/* If the user selects dynamic rules and specifies the algorithm to
 * use, then this function is called.  */
int ompi_coll_tuned_neighbor_alltoallv_intra_do_this(const void *sbuf, const int *scounts, const int *sdisps,
                                                      struct ompi_datatype_t *sdtype,
                                                      void* rbuf, const int *rcounts, const int *rdisps,
                                                      struct ompi_datatype_t *rdtype,
                                                      struct ompi_communicator_t *comm,
                                                      mca_coll_base_module_t *module,
                                                      int algorithm)
{
    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "coll:tuned:neighbor_alltoallv_intra_do_this selected algorithm %d ",
                 algorithm));
    switch (algorithm) {
    case (0):
        return ompi_coll_tuned_neighbor_alltoallv_intra_dec_fixed(sbuf, scounts, sdisps, sdtype,
                                                                  rbuf, rcounts, rdisps, rdtype,
                                                                  comm, module);
    case (1):
        return ompi_coll_base_neighbor_alltoallv(sbuf, scounts, sdisps, sdtype,
                                                 rbuf, rcounts, rdisps, rdtype,
                                                 comm, module);
    case (2):
        return ompi_coll_base_neighbor_alltoallv_controlled(sbuf, scounts, sdisps, sdtype,
                                                            rbuf, rcounts, rdisps, rdtype,
                                                            comm, module, coll_tuned_neighbor_alltoallv_controlled_limit);
    }  /* switch */
    OPAL_OUTPUT((ompi_coll_tuned_stream,
                 "coll:tuned:neighbor_alltoallv_intra_do_this attempt to select "
                 "algorithm %d when only 0-%d is valid.",
                 algorithm, ompi_coll_tuned_forced_max_algorithms[NEIGHBOR_ALLTOALLV]));
    return (MPI_ERR_ARG);
}
