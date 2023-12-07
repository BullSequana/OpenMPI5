/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2016-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2017      Ian Bradley Morgan and Anthony Skjellum. All
 *                         rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "coll_bullnbc.h"
#include "coll_bullnbc_internal.h"

#include "mpi.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/base/coll_base_dynamic_file.h"
#include "opal/util/show_help.h"
#include "coll_bullnbc_partitioned.h"

/*
 * Public string showing the coll ompi_bullnbc component version number
 */
const char *mca_coll_bullnbc_component_version_string =
    "Open MPI bullnbc collective MCA component version " OMPI_VERSION;


static int bullnbc_priority = 10;
static bool bullnbc_in_progress = false;     /* protect from recursive calls */
bool bullnbc_ibcast_skip_dt_decision = true;


static int bullnbc_open(void);
static int bullnbc_close(void);
static int bullnbc_register(void);
static int bullnbc_init_query(bool, bool);
static mca_coll_base_module_t *bullnbc_comm_query(struct ompi_communicator_t *, int *);
static int bullnbc_module_enable(mca_coll_base_module_t *, struct ompi_communicator_t *);

OBJ_CLASS_INSTANCE(ompi_coll_bullnbc_pcoll_request_t,
                   ompi_request_t,
                   NULL,
                   NULL);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

ompi_coll_bullnbc_component_t mca_coll_bullnbc_component = {
    {
        /* First, the mca_component_t struct containing meta information
         * about the component itself */
        .collm_version = {
            MCA_COLL_BASE_VERSION_2_4_0,

            /* Component name and version */
            .mca_component_name = "bullnbc",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),

            /* Component open and close functions */
            .mca_open_component = bullnbc_open,
            .mca_close_component = bullnbc_close,
            .mca_register_component_params = bullnbc_register,
        },
        .collm_data = {
            /* The component is checkpoint ready */
            MCA_BASE_METADATA_PARAM_CHECKPOINT
        },

        /* Initialization / querying functions */
        .collm_init_query = bullnbc_init_query,
        .collm_comm_query = bullnbc_comm_query,
    }
};

static int
bullnbc_open(void)
{
    int ret;
    if (mca_coll_bullnbc_component.dynamic_rules_verbose > 0) {
        mca_coll_bullnbc_component.stream = opal_output_open(NULL);
        opal_output_set_verbosity(mca_coll_bullnbc_component.stream, mca_coll_bullnbc_component.dynamic_rules_verbose);
    } else {
        mca_coll_bullnbc_component.stream = -1;
    }
    if(mca_coll_bullnbc_component.dynamic_rules_filename ) {
        int rc;
        opal_output_verbose(10, mca_coll_bullnbc_component.stream ,
            "coll:bullnbc:component_open Reading collective rules file [%s] which format is %d",
                     mca_coll_bullnbc_component.dynamic_rules_filename,
                     mca_coll_bullnbc_component.dynamic_rules_fileformat);
        rc = ompi_coll_base_read_rules_config_file( mca_coll_bullnbc_component.dynamic_rules_filename,
                                                    mca_coll_bullnbc_component.dynamic_rules_fileformat,
                                                    &(mca_coll_bullnbc_component.all_base_rules), COLLCOUNT);
        if( rc >= 0 ) {
            opal_output_verbose(10, mca_coll_bullnbc_component.stream ,"coll:bullnbc:module_open Read %d valid rules\n", rc);
            if(ompi_coll_base_framework.framework_verbose >= 50) {
                ompi_coll_base_dump_all_rules (mca_coll_bullnbc_component.all_base_rules, COLLCOUNT);
            }
        } else {
            opal_output_verbose(1, mca_coll_bullnbc_component.stream ,"coll:bullnbc:module_open Reading collective rules file failed\n");
            unsigned error_name_len = 12;
            char error_name[error_name_len];
            snprintf(error_name, error_name_len, "file fail%1d", rc);
            error_name[error_name_len - 1] = '\0';
            opal_show_help("help-mpi-coll-bullnbc.txt", (const char*)error_name, true,
                           mca_coll_bullnbc_component.dynamic_rules_filename, mca_coll_bullnbc_component.dynamic_rules_fileformat);
            mca_coll_bullnbc_component.all_base_rules = NULL;
        }
    } else {
        mca_coll_bullnbc_component.all_base_rules = NULL;
    }

    coll_bullnbc_init_part_dag();

    OBJ_CONSTRUCT(&mca_coll_bullnbc_component.active_requests, opal_list_t);

    OBJ_CONSTRUCT(&subpart_free_list, opal_free_list_t);
    OBJ_CONSTRUCT(&mca_coll_bullnbc_component.lock, opal_mutex_t);
    OBJ_CONSTRUCT(&mca_coll_bullnbc_component.requests, opal_free_list_t);
    OBJ_CONSTRUCT(&mca_coll_bullnbc_component.schedules, opal_free_list_t);

    OBJ_CONSTRUCT(&mca_coll_bullnbc_pcoll_requests, opal_free_list_t);
    OBJ_CONSTRUCT(&part_req_free_list, opal_free_list_t);
    OBJ_CONSTRUCT(&pending_pcoll_requests, opal_list_t);
    OBJ_CONSTRUCT(&pcoll_list_lock, opal_mutex_t);
    ret = opal_free_list_init (&part_req_free_list,
                         sizeof(ompi_request_t),
                         opal_cache_line_size,
                         OBJ_CLASS(ompi_request_t),
                         0,opal_cache_line_size,
                         0, -1, 32,
                         NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(1, ompi_coll_base_framework.framework_output,
                            "Partionned pt2pt free_list init failed in bullnbc: %d \n", ret);
        return ret;
    }
    ret = opal_free_list_init (&subpart_free_list,
                         sizeof(ompi_coll_bullnbc_subpart),
                         opal_cache_line_size,
                         OBJ_CLASS(ompi_coll_bullnbc_subpart),
                         0,opal_cache_line_size,
                         0, -1, 32,
                         NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(1, ompi_coll_base_framework.framework_output,
                            "Free list init of subpart for partitioned failed in bullnbc: %d \n", ret);
        return ret;
    }
    ret = opal_free_list_init (&mca_coll_bullnbc_pcoll_requests,
                         sizeof(ompi_coll_bullnbc_pcoll_request_t),
                         opal_cache_line_size,
                         OBJ_CLASS(ompi_coll_bullnbc_pcoll_request_t),
                         0,opal_cache_line_size,
                         0, -1, 32,
                         NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(1, ompi_coll_base_framework.framework_output,
                            "DAG partionned free_list init failed in bullnbc: %d \n", ret);
        return ret;
    }
    ret = opal_free_list_init (&mca_coll_bullnbc_component.requests,
                               sizeof(ompi_coll_bullnbc_request_t), 8,
                               OBJ_CLASS(ompi_coll_bullnbc_request_t),
                               0, 0, 0, -1, 8, NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(1, ompi_coll_base_framework.framework_output,
                            "Request free_list init failed in bullnbc: %d \n", ret);
        return ret;
    }
    if (OMPI_SUCCESS != ret) return ret;
    ret = opal_free_list_init (&mca_coll_bullnbc_component.schedules,
                               sizeof(BULLNBC_Schedule), opal_cache_line_size,
                               OBJ_CLASS(BULLNBC_Schedule),
                               0, opal_cache_line_size,
                               0, -1, 32,
                               NULL, 0, NULL, NULL, NULL);
    if (OMPI_SUCCESS != ret) {
        opal_output_verbose(1, ompi_coll_base_framework.framework_output, "Free_list  failed bullnbc %d \n", ret);
        return ret;
    }
    if (OMPI_SUCCESS != ret) return ret;

    /* note: active comms is the number of communicators who have had
       a non-blocking collective started */
    mca_coll_bullnbc_component.active_comms = 0;

    return OMPI_SUCCESS;
}

static int
bullnbc_close(void)
{
    if (0 != mca_coll_bullnbc_component.active_comms) {
        opal_progress_unregister(ompi_coll_bullnbc_progress);
    }

    coll_bullnbc_fini_part_dag();

    OBJ_DESTRUCT(&mca_coll_bullnbc_pcoll_requests);
    OBJ_DESTRUCT(&pending_pcoll_requests);
    OBJ_DESTRUCT(&pcoll_list_lock);
    OBJ_DESTRUCT(&subpart_free_list);
    OBJ_DESTRUCT(&part_req_free_list);

    OBJ_DESTRUCT(&mca_coll_bullnbc_component.requests);
    OBJ_DESTRUCT(&mca_coll_bullnbc_component.schedules);
    OBJ_DESTRUCT(&mca_coll_bullnbc_component.active_requests);
    OBJ_DESTRUCT(&mca_coll_bullnbc_component.lock);

    if( NULL != mca_coll_bullnbc_component.all_base_rules ) {
        ompi_coll_base_free_all_rules(mca_coll_bullnbc_component.all_base_rules, COLLCOUNT);
        mca_coll_bullnbc_component.all_base_rules = NULL;
    }
    /* close stream */
    if(mca_coll_bullnbc_component.stream >= 0) {
        opal_output_close(mca_coll_bullnbc_component.stream);
    }
    return OMPI_SUCCESS;
}


static int
bullnbc_register(void)
{
    /* Disabled by default */
    bullnbc_priority = 0;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "priority", "Priority of the bullnbc coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &bullnbc_priority);

    /* ibcast decision function can make the wrong decision if a legal
     * non-uniform data type signature is used. This has resulted in the
     * collective operation failing, and possibly producing wrong answers.
     * We are investigating a fix for this problem, but it is taking a while.
     *   https://github.com/open-mpi/ompi/issues/2256
     *   https://github.com/open-mpi/ompi/issues/1763
     * As a result we are adding an MCA parameter to make a conservative
     * decision to avoid this issue. If the user knows that their application
     * does not use data types in this way, then they can set this parameter
     * to get the old behavior. Once the issue is truely fixed, then this
     * parameter can be removed.
     */
    bullnbc_ibcast_skip_dt_decision = true;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "ibcast_skip_dt_decision",
                                           "In ibcast only use size of communicator to choose algorithm, exclude data type signature. Set to 'false' to use data type signature in decision. WARNING: If you set this to 'false' then your application should not use non-uniform data type signatures in calls to ibcast.",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &bullnbc_ibcast_skip_dt_decision);

    mca_coll_bullnbc_component.debug_verbose = 0;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version, "debug_verbose",
                                           "Verbose level of the debug in bullnbc coll component."
                                           " Examples: 0: no verbose, 5: executed cmds, 10: cmds appended",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.debug_verbose);

    mca_coll_bullnbc_component.dynamic_rules_verbose = 0;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version, "dynamic_rules_verbose",
                                           "Verbose level of the bullnbc coll component regarding on dynamic rules."
                                           " Examples: 0: no verbose, 1: selection errors, 10: selection output",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.dynamic_rules_verbose);

    mca_coll_bullnbc_component.use_dynamic_rules = false;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "use_dynamic_rules",
                                           "Switch used to decide if we use static (compiled/if statements) or dynamic (built at runtime) decision function rules (intra communicator only)",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.use_dynamic_rules);
                                           
    mca_coll_bullnbc_component.dynamic_rules_filename = NULL;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "dynamic_rules_filename",
                                           "Filename of configuration file that contains the dynamic (@runtime) decision function rules",
                                           MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.dynamic_rules_filename);

    mca_coll_bullnbc_component.dynamic_rules_fileformat = 0;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "dynamic_rules_fileformat",
                                           "Format of configuration file that "
                                           "contains the dynamic (@runtime) "
                                           "decision function rules. "
                                           "Accepted values are: "
                                           "0 <comm_size, msg_size>, "
                                           "1 <nodes_nb, comm_size, msg_size>, "
                                           "2 <nodes_nb, comm_size, msg_size> with max requests",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.dynamic_rules_fileformat);
    /* Add a MCA parameter to enable/disable discarding of algorithm in case of non commutative operations.
     * When algorithm configuration file (including reduce definition) or reduce_algorithm MCA parameter are used
     * loaded algorithms may not support non commutative operations. Consequently, an issue happens when
     * the operation argument of the MPI_Reduce call is non commutative. To avoid this strong limitation, we provide
     * a discarding mechanism on top of algorithm selection to force the use of a fallback algorithm. This mechanism
     * can be enable/disable using the following MCA parameter. */
    mca_coll_bullnbc_component.reduce_allow_non_commutative_support = false;
    (void) mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                           "reduce_allow_non_commutative_support",
                                           "Switch to allow non commutative operations in ireduce/iallreduce algorithms designed for only commutative operations. Be carefull, enabling this parameter may lead to erroneous numerical results.",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_5,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &mca_coll_bullnbc_component.reduce_allow_non_commutative_support);

    mca_coll_bullnbc_component.debug_read_user_data = false;
    mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                    "debug_read_user_data",
                                    "DEBUG: bullnbc read user data in debug prints",
                                    MCA_BASE_VAR_TYPE_BOOL,
                                    NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                    OPAL_INFO_LVL_5,
                                    MCA_BASE_VAR_SCOPE_ALL,
                                    &mca_coll_bullnbc_component.debug_read_user_data);

    ompi_coll_bullnbc_allgather_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLGATHER]);
    ompi_coll_bullnbc_allgatherv_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLGATHERV]);
    ompi_coll_bullnbc_allreduce_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLREDUCE]);
    ompi_coll_bullnbc_alltoall_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLTOALL]);
    ompi_coll_bullnbc_alltoallv_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLTOALLV]);
    ompi_coll_bullnbc_alltoallw_check_forced_init (&mca_coll_bullnbc_component.forced_params[ALLTOALLW]);
    ompi_coll_bullnbc_barrier_check_forced_init (&mca_coll_bullnbc_component.forced_params[BARRIER]);
    ompi_coll_bullnbc_bcast_check_forced_init (&mca_coll_bullnbc_component.forced_params[BCAST]);
    ompi_coll_bullnbc_exscan_check_forced_init (&mca_coll_bullnbc_component.forced_params[EXSCAN]);
    ompi_coll_bullnbc_gather_check_forced_init (&mca_coll_bullnbc_component.forced_params[GATHER]);
    ompi_coll_bullnbc_gatherv_check_forced_init (&mca_coll_bullnbc_component.forced_params[GATHERV]);
    ompi_coll_bullnbc_gatherw_check_forced_init (&mca_coll_bullnbc_component.forced_params[GATHERW]);
    ompi_coll_bullnbc_reduce_check_forced_init (&mca_coll_bullnbc_component.forced_params[REDUCE]);
    ompi_coll_bullnbc_reduce_scatter_check_forced_init (&mca_coll_bullnbc_component.forced_params[REDUCESCATTER]);
    ompi_coll_bullnbc_reduce_scatter_block_check_forced_init (&mca_coll_bullnbc_component.forced_params[REDUCESCATTERBLOCK]);
    ompi_coll_bullnbc_scan_check_forced_init (&mca_coll_bullnbc_component.forced_params[SCAN]);
    ompi_coll_bullnbc_scatter_check_forced_init (&mca_coll_bullnbc_component.forced_params[SCATTER]);
    ompi_coll_bullnbc_scatterv_check_forced_init (&mca_coll_bullnbc_component.forced_params[SCATTERV]);
    ompi_coll_bullnbc_neighbor_allgather_check_forced_init (&mca_coll_bullnbc_component.forced_params[NEIGHBOR_ALLGATHER]);
    ompi_coll_bullnbc_neighbor_allgatherv_check_forced_init (&mca_coll_bullnbc_component.forced_params[NEIGHBOR_ALLGATHERV]);
    ompi_coll_bullnbc_neighbor_alltoall_check_forced_init (&mca_coll_bullnbc_component.forced_params[NEIGHBOR_ALLTOALL]);
    ompi_coll_bullnbc_neighbor_alltoallv_check_forced_init (&mca_coll_bullnbc_component.forced_params[NEIGHBOR_ALLTOALLV]);
    ompi_coll_bullnbc_neighbor_alltoallw_check_forced_init (&mca_coll_bullnbc_component.forced_params[NEIGHBOR_ALLTOALLW]);

    bullnbc_partitioned_register();

    return OMPI_SUCCESS;
}



/*
 * Initial query function that is invoked during MPI_INIT, allowing
 * this component to disqualify itself if it doesn't support the
 * required level of thread support.
 */
static int
bullnbc_init_query(bool enable_progress_threads,
                  bool enable_mpi_threads)
{
    /* Nothing to do */
    return OMPI_SUCCESS;
}

/*
 * Invoked when there's a new communicator that has been created.
 * Look at the communicator and decide which set of functions and
 * priority we want to return.
 */
mca_coll_base_module_t *
bullnbc_comm_query(struct ompi_communicator_t *comm,
                  int *priority)
{
    ompi_coll_bullnbc_module_t *module;

    module = OBJ_NEW(ompi_coll_bullnbc_module_t);
    if (NULL == module) return NULL;

    *priority = bullnbc_priority;

    module->super.coll_module_enable = bullnbc_module_enable;
    if (OMPI_COMM_IS_INTER(comm)) {
        module->super.coll_iallgather = ompi_coll_bullnbc_iallgather_inter;
        module->super.coll_iallgatherv = ompi_coll_bullnbc_iallgatherv_inter;
        module->super.coll_iallreduce = ompi_coll_bullnbc_iallreduce_inter;
        module->super.coll_ialltoall = ompi_coll_bullnbc_ialltoall_inter;
        module->super.coll_ialltoallv = ompi_coll_bullnbc_ialltoallv_inter;
        module->super.coll_ialltoallw = ompi_coll_bullnbc_ialltoallw_inter;
        module->super.coll_ibarrier = ompi_coll_bullnbc_ibarrier_inter;
        module->super.coll_ibcast = ompi_coll_bullnbc_ibcast_inter;
        module->super.coll_iexscan = NULL;
        module->super.coll_igather = ompi_coll_bullnbc_igather_inter;
        module->super.coll_igatherv = ompi_coll_bullnbc_igatherv_inter;
        module->super.coll_igatherw = ompi_coll_bullnbc_igatherw_inter;
        module->super.coll_ireduce = ompi_coll_bullnbc_ireduce_inter;
        module->super.coll_ireduce_scatter = ompi_coll_bullnbc_ireduce_scatter_inter;
        module->super.coll_ireduce_scatter_block = ompi_coll_bullnbc_ireduce_scatter_block_inter;
        module->super.coll_iscan = NULL;
        module->super.coll_iscatter = ompi_coll_bullnbc_iscatter_inter;
        module->super.coll_iscatterv = ompi_coll_bullnbc_iscatterv_inter;

        module->super.coll_allgather_init = ompi_coll_bullnbc_allgather_inter_init;
        module->super.coll_allgatherv_init = ompi_coll_bullnbc_allgatherv_inter_init;
        module->super.coll_allreduce_init = ompi_coll_bullnbc_allreduce_inter_init;
        module->super.coll_alltoall_init = ompi_coll_bullnbc_alltoall_inter_init;
        module->super.coll_alltoallv_init = ompi_coll_bullnbc_alltoallv_inter_init;
        module->super.coll_alltoallw_init = ompi_coll_bullnbc_alltoallw_inter_init;
        module->super.coll_barrier_init = ompi_coll_bullnbc_barrier_inter_init;
        module->super.coll_bcast_init = ompi_coll_bullnbc_bcast_inter_init;
        module->super.coll_exscan_init = NULL;
        module->super.coll_gather_init = ompi_coll_bullnbc_gather_inter_init;
        module->super.coll_gatherv_init = ompi_coll_bullnbc_gatherv_inter_init;
        module->super.coll_gatherw_init = ompi_coll_bullnbc_gatherw_inter_init;
        module->super.coll_reduce_init = ompi_coll_bullnbc_reduce_inter_init;
        module->super.coll_reduce_scatter_init = ompi_coll_bullnbc_reduce_scatter_inter_init;
        module->super.coll_reduce_scatter_block_init = ompi_coll_bullnbc_reduce_scatter_block_inter_init;
        module->super.coll_scan_init = NULL;
        module->super.coll_scatter_init = ompi_coll_bullnbc_scatter_inter_init;
        module->super.coll_scatterv_init = ompi_coll_bullnbc_scatterv_inter_init;
        module->super.coll_scatterw_init = NULL;

#if OMPI_MPI_PARTITIONED_COLL
        module->super.coll_palltoall_init = ompi_coll_bullnbc_palltoall_init;
        module->super.coll_palltoallr_init = ompi_coll_bullnbc_palltoallr_init_select;
        module->super.coll_palltoallv_init = ompi_coll_bullnbc_palltoallv_init;
        module->super.coll_palltoallvr_init = ompi_coll_bullnbc_palltoallvr_init;
        module->super.coll_pbcast_init = ompi_coll_bullnbc_pbcast_init;
        module->super.coll_pbcastr_init = ompi_coll_bullnbc_pbcastr_init;
        module->super.coll_pgather_init = ompi_coll_bullnbc_pgather_init;
        module->super.coll_pgatherr_init = ompi_coll_bullnbc_pgatherr_init;
        module->super.coll_preduce_init = ompi_coll_bullnbc_preduce_init;
        module->super.coll_preducer_init = ompi_coll_bullnbc_preducer_init;
#endif // OMPI_MPI_PARTITIONED_COLL

    } else {
        module->super.coll_iallgather = ompi_coll_bullnbc_iallgather;
        module->super.coll_iallgatherv = ompi_coll_bullnbc_iallgatherv;
        module->super.coll_iallreduce = ompi_coll_bullnbc_iallreduce;
        module->super.coll_ialltoall = ompi_coll_bullnbc_ialltoall;
        module->super.coll_ialltoallv = ompi_coll_bullnbc_ialltoallv;
        module->super.coll_ialltoallw = ompi_coll_bullnbc_ialltoallw;
        module->super.coll_ibarrier = ompi_coll_bullnbc_ibarrier;
        module->super.coll_ibcast = ompi_coll_bullnbc_ibcast;
        module->super.coll_iexscan = ompi_coll_bullnbc_iexscan;
        module->super.coll_igather = ompi_coll_bullnbc_igather;
        module->super.coll_igatherv = ompi_coll_bullnbc_igatherv;
        module->super.coll_igatherw = ompi_coll_bullnbc_igatherw;
        module->super.coll_ireduce = ompi_coll_bullnbc_ireduce;
        module->super.coll_ireduce_scatter = ompi_coll_bullnbc_ireduce_scatter;
        module->super.coll_ireduce_scatter_block = ompi_coll_bullnbc_ireduce_scatter_block;
        module->super.coll_iscan = ompi_coll_bullnbc_iscan;
        module->super.coll_iscatter = ompi_coll_bullnbc_iscatter;
        module->super.coll_iscatterv = ompi_coll_bullnbc_iscatterv;

        module->super.coll_ineighbor_allgather = ompi_coll_bullnbc_ineighbor_allgather;
        module->super.coll_ineighbor_allgatherv = ompi_coll_bullnbc_ineighbor_allgatherv;
        module->super.coll_ineighbor_alltoall = ompi_coll_bullnbc_ineighbor_alltoall;
        module->super.coll_ineighbor_alltoallv = ompi_coll_bullnbc_ineighbor_alltoallv;
        module->super.coll_ineighbor_alltoallw = ompi_coll_bullnbc_ineighbor_alltoallw;

        module->super.coll_allgather_init = ompi_coll_bullnbc_allgather_init;
        module->super.coll_allgatherv_init = ompi_coll_bullnbc_allgatherv_init;
        module->super.coll_allreduce_init = ompi_coll_bullnbc_allreduce_init;
        module->super.coll_alltoall_init = ompi_coll_bullnbc_alltoall_init;
        module->super.coll_alltoallv_init = ompi_coll_bullnbc_alltoallv_init;
        module->super.coll_alltoallw_init = ompi_coll_bullnbc_alltoallw_init;
        module->super.coll_barrier_init = ompi_coll_bullnbc_barrier_init;
        module->super.coll_bcast_init = ompi_coll_bullnbc_bcast_init;
        module->super.coll_exscan_init = ompi_coll_bullnbc_exscan_init;
        module->super.coll_gather_init = ompi_coll_bullnbc_gather_init;
        module->super.coll_gatherv_init = ompi_coll_bullnbc_gatherv_init;
        module->super.coll_gatherw_init = ompi_coll_bullnbc_gatherw_init;
        module->super.coll_reduce_init = ompi_coll_bullnbc_reduce_init;
        module->super.coll_reduce_scatter_init = ompi_coll_bullnbc_reduce_scatter_init;
        module->super.coll_reduce_scatter_block_init = ompi_coll_bullnbc_reduce_scatter_block_init;
        module->super.coll_scan_init = ompi_coll_bullnbc_scan_init;
        module->super.coll_scatter_init = ompi_coll_bullnbc_scatter_init;
        module->super.coll_scatterv_init = ompi_coll_bullnbc_scatterv_init;
        module->super.coll_scatterw_init = NULL;

#if OMPI_MPI_PARTITIONED_COLL
        module->super.coll_palltoall_init = ompi_coll_bullnbc_palltoall_init;
        module->super.coll_palltoallr_init = ompi_coll_bullnbc_palltoallr_init_select;
        module->super.coll_palltoallv_init = ompi_coll_bullnbc_palltoallv_init;
        module->super.coll_palltoallvr_init = ompi_coll_bullnbc_palltoallvr_init;
        module->super.coll_pbcast_init = ompi_coll_bullnbc_pbcast_init;
        module->super.coll_pbcastr_init = ompi_coll_bullnbc_pbcastr_init;
        module->super.coll_pgather_init = ompi_coll_bullnbc_pgather_init;
        module->super.coll_pgatherr_init = ompi_coll_bullnbc_pgatherr_init;
        module->super.coll_preduce_init = ompi_coll_bullnbc_preduce_init;
        module->super.coll_preducer_init = ompi_coll_bullnbc_preducer_init;
#endif // OMPI_MPI_PARTITIONED_COLL

        module->super.coll_neighbor_allgather_init = ompi_coll_bullnbc_neighbor_allgather_init;
        module->super.coll_neighbor_allgatherv_init = ompi_coll_bullnbc_neighbor_allgatherv_init;
        module->super.coll_neighbor_alltoall_init = ompi_coll_bullnbc_neighbor_alltoall_init;
        module->super.coll_neighbor_alltoallv_init = ompi_coll_bullnbc_neighbor_alltoallv_init;
        module->super.coll_neighbor_alltoallw_init = ompi_coll_bullnbc_neighbor_alltoallw_init;
    }

    module->super.coll_agree = NULL;
    module->super.coll_iagree = NULL;

    if (OMPI_SUCCESS != BULLNBC_Init_comm(comm, module)) {
        OBJ_RELEASE(module);
        return NULL;
    }

    return &(module->super);
}


/*
 * Init module on the communicator
 */
static int
bullnbc_module_enable(mca_coll_base_module_t *module,
                     struct ompi_communicator_t *comm)
{
  ompi_coll_bullnbc_module_t* nbc_module = (ompi_coll_bullnbc_module_t*) module;
  int i;
  if(mca_coll_bullnbc_component.use_dynamic_rules && mca_coll_bullnbc_component.all_base_rules) {
    int size, nnodes;
    /* Allocate the data that hangs off the communicator */
    if (OMPI_COMM_IS_INTER(comm)) {
      size = ompi_comm_remote_size(comm);
    } else {
      size = ompi_comm_size(comm);
    }
    /* Get the number of nodes in communicator */
    nnodes = ompi_coll_base_get_nnodes(comm);
    for(i=0;i<COLLCOUNT;i++) {
        nbc_module->com_rules[i] = ompi_coll_base_get_com_rule_ptr(mca_coll_bullnbc_component.all_base_rules,
                                                                   i, nnodes, size );
    }
  } else {
    for(i=0;i<COLLCOUNT;i++) {
        nbc_module->com_rules[i] = NULL;
    }
  }
    /* All done */
    return OMPI_SUCCESS;
}


static int
ompi_coll_bullnbc_pcoll_progress(void)
{
    ompi_coll_bullnbc_pcoll_request_t *pcoll_req, *next_pcoll_req;
    int count = 0;
    static int in_progress = 0;
    if (in_progress) {
        /* Removing many items to lists during this progress may corrupt
         * parent progress list traversal */
        return count;
    }
    in_progress = 1;

    OPAL_THREAD_LOCK(&pcoll_list_lock);
    OPAL_LIST_FOREACH_SAFE(pcoll_req, next_pcoll_req, &pending_pcoll_requests,
                           ompi_coll_bullnbc_pcoll_request_t){

        count += pcoll_req->progress_cb(pcoll_req);
    }
    OPAL_THREAD_UNLOCK(&pcoll_list_lock);

    in_progress = 0;
    return count;
}

int
ompi_coll_bullnbc_progress(void)
{
    ompi_coll_bullnbc_request_t* request;
    int completed = 0;

    /* Partitoned collectives progress */
    completed += ompi_coll_bullnbc_pcoll_progress();

    if (0 == opal_list_get_size (&mca_coll_bullnbc_component.active_requests)) {
        /* no requests -- nothing to do. do not grab a lock */
        return 0;
    }

    /* process active requests, and use mca_coll_bullnbc_component.lock to access the
     * mca_coll_bullnbc_component.active_requests list */
    OPAL_THREAD_LOCK(&mca_coll_bullnbc_component.lock);
    /* return if invoked recursively */
    if (!bullnbc_in_progress) {
        ompi_coll_bullnbc_request_t *next;
        bullnbc_in_progress = true;

        OPAL_LIST_FOREACH_SAFE(request, next, &mca_coll_bullnbc_component.active_requests,
                               ompi_coll_bullnbc_request_t) {
            int res;
            OPAL_THREAD_UNLOCK(&mca_coll_bullnbc_component.lock);
            res = BULLNBC_Progress(request);
            if( NBC_CONTINUE != res ) {
                /* done, remove and complete */
                OPAL_THREAD_LOCK(&mca_coll_bullnbc_component.lock);
                opal_list_remove_item(&mca_coll_bullnbc_component.active_requests,
                                      &request->super.super.super.super);
                OPAL_THREAD_UNLOCK(&mca_coll_bullnbc_component.lock);

                if (OMPI_SUCCESS == res) {
                    request->super.super.req_status.MPI_ERROR = OMPI_SUCCESS;
                } else {
                    request->super.super.req_status.MPI_ERROR = res;
                }
                if(request->super.super.req_persistent) {
                    /* reset for the next communication */
                    request->row_offset = 0;
                }
                if(!request->super.super.req_persistent || !REQUEST_COMPLETE(&request->super.super)) {
            	    ompi_request_complete(&request->super.super, true);
                }
                completed++;
            }
            OPAL_THREAD_LOCK(&mca_coll_bullnbc_component.lock);
        }

        bullnbc_in_progress = false;
    }
    OPAL_THREAD_UNLOCK(&mca_coll_bullnbc_component.lock);

    return completed;
}


static void
bullnbc_module_construct(ompi_coll_bullnbc_module_t *module)
{
    OBJ_CONSTRUCT(&module->mutex, opal_mutex_t);
    module->comm_registered = 0;
}


static void
bullnbc_module_destruct(ompi_coll_bullnbc_module_t *module)
{
    OBJ_DESTRUCT(&module->mutex);

    /* if we ever were used for a collective op, do the progress cleanup. */
    if (1 == module->comm_registered) {
        int32_t tmp =
            OPAL_THREAD_ADD_FETCH32(&mca_coll_bullnbc_component.active_comms, -1);
        if (0 == tmp) {
            opal_progress_unregister(ompi_coll_bullnbc_progress);
        }
    }
}


OBJ_CLASS_INSTANCE(ompi_coll_bullnbc_subpart,
                   opal_free_list_item_t,
                   NULL,
                   NULL);
OBJ_CLASS_INSTANCE(ompi_coll_bullnbc_module_t,
                   mca_coll_base_module_t,
                   bullnbc_module_construct,
                   bullnbc_module_destruct);


static int
request_start(size_t count, ompi_request_t ** requests)
{
    size_t i;

    NBC_DEBUG(5, " ** request_start **\n");

    for (i = 0; i < count; i++) {
        int res;
        NBC_Handle *handle = (NBC_Handle *) requests[i];
        BULLNBC_Schedule *schedule = handle->schedule;

        NBC_DEBUG(5, "--------------------------------\n");
        NBC_DEBUG(5, "schedule %p size %u\n", &schedule, sizeof(schedule));
        NBC_DEBUG(5, "handle %p size %u\n", &handle, sizeof(handle));
        NBC_DEBUG(5, "req_array %p size %u\n", &handle->req_array, sizeof(handle->req_array));
        NBC_DEBUG(5, "row_offset=%u address=%p size=%u\n", handle->row_offset, &handle->row_offset, sizeof(handle->row_offset));
        NBC_DEBUG(5, "req_count=%u address=%p size=%u\n", handle->req_count, &handle->req_count, sizeof(handle->req_count));
        NBC_DEBUG(5, "tmpbuf address=%p size=%u\n", handle->tmpbuf, sizeof(handle->tmpbuf));
        NBC_DEBUG(5, "--------------------------------\n");

        handle->super.super.req_complete = REQUEST_PENDING;
        handle->nbc_complete = false;

        res = NBC_Start(handle);
        if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
            NBC_DEBUG(5, " ** bad result from NBC_Start **\n");
            return res;
        }
    }

    NBC_DEBUG(5, " ** LEAVING request_start **\n");

    return OMPI_SUCCESS;

}


static int
request_cancel(struct ompi_request_t *request, int complete)
{
    return MPI_ERR_REQUEST;
}


static int
request_free(struct ompi_request_t **ompi_req)
{
    ompi_coll_bullnbc_request_t *request =
        (ompi_coll_bullnbc_request_t*) *ompi_req;

    if( !REQUEST_COMPLETE(&request->super.super) ) {
        return MPI_ERR_REQUEST;
    }

    OMPI_COLL_BULLNBC_REQUEST_RETURN(request);
    *ompi_req = MPI_REQUEST_NULL;

    return OMPI_SUCCESS;
}


static void
request_construct(ompi_coll_bullnbc_request_t *request)
{
    request->super.super.req_type = OMPI_REQUEST_COLL;
    request->super.super.req_status._cancelled = 0;
    request->super.super.req_start = request_start;
    request->super.super.req_free = request_free;
    request->super.super.req_cancel = request_cancel;
}


OBJ_CLASS_INSTANCE(ompi_coll_bullnbc_request_t,
                   ompi_coll_base_nbc_request_t,
                   request_construct,
                   NULL);
