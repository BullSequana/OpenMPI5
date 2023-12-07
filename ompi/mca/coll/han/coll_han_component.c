/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
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
 * Most of the description of the data layout is in the
 * coll_han_module.c file.
 */

#include "ompi_config.h"

#include "opal/util/show_help.h"
#include "opal/util/argv.h"
#include "ompi/constants.h"
#include "ompi/mca/coll/coll.h"
#include "coll_han.h"
#include "coll_han_dynamic.h"
#include "coll_han_dynamic_file.h"
#include "coll_han_algorithms.h"
#include "ompi/mca/coll/base/coll_base_util.h"

/*
 * Public string showing the coll ompi_han component version number
 */
const char *mca_coll_han_component_version_string =
    "Open MPI HAN collective MCA component version " OMPI_VERSION;

ompi_coll_han_components ompi_coll_han_available_components[COMPONENTS_COUNT] = {
    { SELF, "self",  NULL },
    { BASIC, "basic", NULL },
    { LIBNBC, "libnbc", NULL },
    { TUNED, "tuned", NULL },
    { SM, "sm", NULL },
    { BSHARED, "bshared", NULL },
    { ADAPT, "adapt", NULL },
    { HAN, "han", NULL }
};

/*
 * Local functions
 */
static int han_open(void);
static int han_close(void);
static int han_register(void);

/*
 * Instantiate the public struct with all of our public information
 * and pointers to our public functions in it
 */

mca_coll_han_component_t mca_coll_han_component = {
    /* First, fill in the super */
    {
        /* First, the mca_component_t struct containing meta
           information about the component itself */

        .collm_version = {
            MCA_COLL_BASE_VERSION_2_4_0,

            /* Component name and version */
            .mca_component_name = "han",
            MCA_BASE_MAKE_VERSION(component, OMPI_MAJOR_VERSION, OMPI_MINOR_VERSION,
                                  OMPI_RELEASE_VERSION),

            /* Component functions */
            .mca_open_component = han_open,
            .mca_close_component = han_close,
            .mca_register_component_params = han_register,
        },
        .collm_data = {
            /* The component is not checkpoint ready */
            MCA_BASE_METADATA_PARAM_NONE},

        /* Initialization / querying functions */

        .collm_init_query = mca_coll_han_init_query,
        .collm_comm_query = mca_coll_han_comm_query,
    },

    /* han-component specific information */

    /* (default) priority */
    .han_priority = 35,
    /* workaround for nvcc compiler */
    .dynamic_rules_filename = NULL,
};

/*
 * Init the component
 */
static int han_open(void)
{
    /* Get the global coll verbosity: it will be ours */
    if (mca_coll_han_component.han_output_verbose) {
        mca_coll_han_component.han_output = opal_output_open(NULL);
        opal_output_set_verbosity(mca_coll_han_component.han_output,
                                  mca_coll_han_component.han_output_verbose);
    } else {
        mca_coll_han_component.han_output = ompi_coll_base_framework.framework_output;
    }

    // mca_coll_han_init_algos() done in register, as enum needed in MCA var

    return mca_coll_han_init_dynamic_rules();
}

/*
 * Shut down the component
 */
static int han_close(void)
{
    mca_coll_han_free_dynamic_rules();

    free(mca_coll_han_component.han_op_module_name.bcast.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.bcast.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.bcast.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.bcast.han_op_low_module_name = NULL;

    free(mca_coll_han_component.han_op_module_name.reduce.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.reduce.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.reduce.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.reduce.han_op_low_module_name = NULL;

    free(mca_coll_han_component.han_op_module_name.allreduce.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.allreduce.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.allreduce.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.allreduce.han_op_low_module_name = NULL;

    free(mca_coll_han_component.han_op_module_name.allgather.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.allgather.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.allgather.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.allgather.han_op_low_module_name = NULL;

    free(mca_coll_han_component.han_op_module_name.gather.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.gather.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.gather.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.gather.han_op_low_module_name = NULL;

    free(mca_coll_han_component.han_op_module_name.scatter.han_op_up_module_name);
    mca_coll_han_component.han_op_module_name.scatter.han_op_up_module_name = NULL;
    free(mca_coll_han_component.han_op_module_name.scatter.han_op_low_module_name);
    mca_coll_han_component.han_op_module_name.scatter.han_op_low_module_name = NULL;

    mca_coll_han_free_algorithms();
    if (mca_coll_han_component.han_output_verbose) {
        opal_output_close(mca_coll_han_component.han_output);
    }

    return OMPI_SUCCESS;
}

/*
 * @return true if the collective has a simple version that does not use tasks.
 */
static bool is_simple_implemented(COLLTYPE_T coll)
{
    switch(coll) {
        case ALLGATHER:
        case ALLREDUCE:
        case BCAST:
        case GATHER:
        case REDUCE:
        case SCATTER:
            return true;
        default:
            return false;
    }
}

/**
 * topo level conversions both ways; str <-> id
 * An enum is used for conversions.
 */
mca_base_var_enum_value_t level_enumerator[] = {
    { LEAF_LEVEL, "leaf_level" },
    { INTRA_NODE, "intra_node" },
    { INTER_NODE, "inter_node" },
    { GATEWAY, "gateway" },
    { GLOBAL_COMMUNICATOR, "global_communicator" },
    { 0 }
};

/*
 * Stringifier for topological level
 */
const char* mca_coll_han_topo_lvl_to_str(TOPO_LVL_T topo_lvl_id)
{
    for (int i = 0; level_enumerator[i].string != NULL; i++) {
        if (topo_lvl_id == (TOPO_LVL_T) level_enumerator[i].value) {
            return level_enumerator[i].string;
        }
    }
    return "invalid topologic level";
}
int mca_coll_han_topo_lvl_name_to_id(const char *topo_level_name)
{
    for (int i = 0; level_enumerator[i].string != NULL; i++) {
        if (0 == strcmp(topo_level_name, level_enumerator[i].string)) {
            return i;
        }
    }
    return -1;
}

static int
mca_coll_han_query_module_from_mca(mca_base_component_t* c,
                                   const char* param_name,
                                   const char* param_doc,
                                   int info_level,
                                   uint32_t* module_id,
                                   char** storage)
{
    char *module_name, *endptr = NULL;

    int mod_id = COMPONENTS_COUNT-1;
    mod_id = (*module_id > (uint32_t)mod_id) ? mod_id : (int)*module_id; /* stay in range */

    *storage = ompi_coll_han_available_components[mod_id].component_name;

    (void) mca_base_component_var_register(c, param_name, param_doc,
                                           MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                           info_level,
                                           MCA_BASE_VAR_SCOPE_READONLY, storage);
    module_name = *storage;
    mod_id = strtol(module_name, &endptr, 10);
    if( module_name == endptr ) {  /* no conversion, maybe we got a module name instead */
        /* Convert module name to id */
        mod_id = mca_coll_han_component_name_to_id(module_name);
    }
    /* Keep the module in the range */
    *module_id = (mod_id < 0) ? 0 : mod_id;

    return OMPI_SUCCESS;
}

const char* mca_coll_han_split_lvl_to_str(SPLIT_LVL_T split_lvl)
{
    switch(split_lvl) {
        case SOCKET:
            return "socket";
        case NODE:
            return "node";
        case CLUSTER:
            return "cluster";
        case NB_SPLIT_LVL:
        default:
            return "invalid split level";
    }
}

/*
 * Register MCA params
 */
static int han_register(void)
{
    mca_base_component_t *c = &mca_coll_han_component.super.collm_version;
    mca_coll_han_component_t *cs = &mca_coll_han_component;

    /* Generated parameters name and description */
    char param_name[128], param_desc[256];
    int param_desc_size;
    COLLTYPE_T coll;
    TOPO_LVL_T topo_lvl;
    COMPONENT_T component;

    (void) mca_base_component_var_register(c, "priority", "Priority of the HAN coll component",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_priority);

    cs->han_output_verbose = 0;
    (void) mca_base_component_var_register(c, "verbose", "Verbosity of the HAN coll component (use coll base verbosity if not set)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_output_verbose);

    cs->han_bcast_segsize = 65536;
    (void) mca_base_component_var_register(c, "bcast_segsize",
                                           "segment size for bcast",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_bcast_segsize);

    cs->han_bcast_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "bcast_up_module",
                                              "up level module for bcast, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_bcast_up_module,
                                              &cs->han_op_module_name.bcast.han_op_up_module_name);

    cs->han_bcast_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "bcast_low_module",
                                              "low level module for bcast, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9,
                                              &cs->han_bcast_low_module,
                                              &cs->han_op_module_name.bcast.han_op_low_module_name);

    cs->han_reduce_segsize = 65536;
    (void) mca_base_component_var_register(c, "reduce_segsize",
                                           "segment size for reduce",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_reduce_segsize);

    cs->han_reduce_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "reduce_up_module",
                                              "up level module for allreduce, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_reduce_up_module,
                                              &cs->han_op_module_name.reduce.han_op_up_module_name);

    cs->han_reduce_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "reduce_low_module",
                                              "low level module for allreduce, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9, &cs->han_reduce_low_module,
                                              &cs->han_op_module_name.reduce.han_op_low_module_name);

    cs->han_allreduce_segsize = 65536;
    (void) mca_base_component_var_register(c, "allreduce_segsize",
                                           "segment size for allreduce",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_allreduce_segsize);

    cs->han_allreduce_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "allreduce_up_module",
                                              "up level module for allreduce, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_allreduce_up_module,
                                              &cs->han_op_module_name.allreduce.han_op_up_module_name);

    cs->han_allreduce_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "allreduce_low_module",
                                              "low level module for allreduce, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9, &cs->han_allreduce_low_module,
                                              &cs->han_op_module_name.allreduce.han_op_low_module_name);

    cs->han_allgather_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "allgather_up_module",
                                              "up level module for allgather, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_allgather_up_module,
                                              &cs->han_op_module_name.allgather.han_op_up_module_name);

    cs->han_allgather_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "allgather_low_module",
                                              "low level module for allgather, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9, &cs->han_allgather_low_module,
                                              &cs->han_op_module_name.allgather.han_op_low_module_name);

    cs->han_gather_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "gather_up_module",
                                              "up level module for gather, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_gather_up_module,
                                              &cs->han_op_module_name.gather.han_op_up_module_name);

    cs->han_gather_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "gather_low_module",
                                              "low level module for gather, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9, &cs->han_gather_low_module,
                                              &cs->han_op_module_name.gather.han_op_low_module_name);

    cs->han_scatter_up_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "scatter_up_module",
                                              "up level module for scatter, 0 libnbc, 1 adapt",
                                              OPAL_INFO_LVL_9, &cs->han_scatter_up_module,
                                              &cs->han_op_module_name.scatter.han_op_up_module_name);

    cs->han_scatter_low_module = 0;
    (void) mca_coll_han_query_module_from_mca(c, "scatter_low_module",
                                              "low level module for scatter, 0 tuned, 1 sm",
                                              OPAL_INFO_LVL_9, &cs->han_scatter_low_module,
                                              &cs->han_op_module_name.scatter.han_op_low_module_name);

    cs->han_scatter_handle_reorder_with_copy = false;
    (void) mca_base_component_var_register(c, "scatter_handle_reorder_with_copy",
                                           "Whether we handle the reorder "
                                           "with a recopy in the correct order or "
                                           "leave it to the inferior layers. "
                                           "It is checked only if processes are not in "
                                           "a linear (a.k.a \"block\" in slurm terminology) "
                                           "distribution."
                                           "(It only works with the scatter recursive)",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_scatter_handle_reorder_with_copy);

    cs->han_scatter_memsize = 1024000;
    (void) mca_base_component_var_register(c, "scatter_memsize",
                                           "Segment size for pipelined memcare scatter algorithm"
                                           "Please take into account already reserved memory"
                                           "and last cache maximum memory",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_scatter_memsize);

    cs->han_scatter_segsize = 400000;
    (void) mca_base_component_var_register(c, "scatter_segsize",
                                           "Minimum segment size for pipelined scatter algorithm"
                                           "in the case where we have more than 1 segment"
                                           "In case we'd need more than mx_nb_segments segments"
                                           "the segsize is allowed to be bigger",
                                           MCA_BASE_VAR_TYPE_SIZE_T, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_scatter_segsize);

    cs->han_scatter_max_nb_segments = 8;
    (void) mca_base_component_var_register(c, "scatter_max_nb_segments",
                                           "Maximum number of segments for pipelined scatter algorithm",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_scatter_max_nb_segments);

    cs->han_alltoall_segsize = 1024;
    (void) mca_base_component_var_register(c, "alltoall_segsize",
                                           "segment size for pipelined grid alltoall algorithm",
                                           MCA_BASE_VAR_TYPE_UNSIGNED_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_9,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->han_alltoall_segsize);

    cs->han_reproducible = 0;
    (void) mca_base_component_var_register(c, "reproducible",
                                           "whether we need reproducible results "
                                           "(enabling this disables optimisations using topology)"
                                           "0 disable 1 enable, default 0",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0,
                                           MCA_BASE_VAR_FLAG_SETTABLE,
                                           OPAL_INFO_LVL_3,
                                           MCA_BASE_VAR_SCOPE_ALL_EQ, &cs->han_reproducible);
    /*
     * Han algorithms MCA parameters for each collective.
     * Shows algorithms thanks to enumerator
     */
    if (OMPI_ERROR == mca_coll_han_init_algorithms()) { // needs to be initialised here to show available algorithms
       return OMPI_ERROR;
    }

    mca_base_var_enum_t *new_enum;
    for(coll = 0 ; coll < COLLCOUNT ; coll++) {
        if (!mca_coll_han_is_coll_dynamic_implemented(coll)
            || (0 == mca_coll_han_component.num_available_algorithms[coll])) {
          continue;
        }
        cs->use_algorithm[coll] = 0; // default algorithm is 0
        snprintf(param_name, sizeof(param_name), "use_%s_algorithm",
                 mca_coll_base_colltype_to_str(coll));
        snprintf(param_desc, sizeof(param_desc), "which han algorithm is used for %s",
                 mca_coll_base_colltype_to_str(coll));
        // note: the enumerator is create in mca_coll_han_init_algorithms()
        (void) mca_base_var_enum_create(param_name,
                                        mca_coll_han_component.algorithm_enumerator[coll],
                                        &new_enum);
        cs->use_algorithm_param[coll] = mca_base_component_var_register(c,
                                        param_name,
                                        param_desc,
                                        MCA_BASE_VAR_TYPE_INT, new_enum, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                        OPAL_INFO_LVL_5,
                                        MCA_BASE_VAR_SCOPE_ALL,
                                        &(cs->use_algorithm[coll]));
        OBJ_RELEASE(new_enum);
    }

    /*
     * Simple algorithms MCA parameters :
     * using simple algorithms will just perform hierarchical communications.
     * By default communications are also split into tasks
     * to handle thread noise
     */
    for(coll = 0 ; coll < COLLCOUNT ; coll++) {
        if (coll != GATHER) {
            cs->use_simple_algorithm[coll] = false;
        } else {
            cs->use_simple_algorithm[coll] = true;
        }
        if(is_simple_implemented(coll)) {
            const char *collstr = mca_coll_base_colltype_to_str(coll);
            snprintf(param_name, sizeof(param_name), "use_simple_%s",
                     collstr);
            snprintf(param_desc, sizeof(param_desc), "whether to enable simple algorithm for %s. "
                     "Prefer use_%s_algorithm=simple or configuration file instead.",
                     collstr, collstr);
            mca_base_component_var_register(c, param_name,
                                            param_desc,
                                            MCA_BASE_VAR_TYPE_BOOL, NULL, 0,
                                            MCA_BASE_VAR_FLAG_DEPRECATED,
                                            OPAL_INFO_LVL_5,
                                            MCA_BASE_VAR_SCOPE_READONLY,
                                            &(cs->use_simple_algorithm[coll]));
        }
    }

    /* Noreorder gather switch */
    cs->use_noreorder_gather=0;
    (void) mca_base_component_var_register(c, "use_noreorder_gather",
                                           "enable no reorder gather algorithm. "
                                           "0 disable 1 enable, default 0",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEPRECATED,
                                           OPAL_INFO_LVL_3,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->use_noreorder_gather);

    /* Dynamic rules MCA parameters */
    /* TODO: Find a way to avoid unused entried */
    memset(cs->mca_sub_components, 0,
           COLLCOUNT * (GLOBAL_COMMUNICATOR+1) * sizeof(COMPONENT_T));

    for(coll = 0; coll < COLLCOUNT; coll++) {
        if(!mca_coll_han_is_coll_dynamic_implemented(coll)) {
            continue;
        }
        /*
         * Default values
         */
        for (topo_lvl = 0 ; topo_lvl < GLOBAL_COMMUNICATOR ; topo_lvl++) {
            cs->mca_sub_components[coll][topo_lvl] = TUNED;
        }
        cs->mca_sub_components[coll][GLOBAL_COMMUNICATOR] = HAN;
        cs->mca_sub_components[coll][LEAF_LEVEL] = TUNED;
        cs->mca_sub_components[coll][INTER_NODE] = BASIC;
    }
    /* Specific default values */
    cs->mca_sub_components[BARRIER][INTRA_NODE] = BSHARED;
    cs->mca_sub_components[BARRIER][INTER_NODE] = TUNED;
    cs->mca_sub_components[BARRIER][LEAF_LEVEL] = BSHARED;

    /* Dynamic rule MCA var registration */
    for(coll = 0; coll < COLLCOUNT; coll++) {
        if(!mca_coll_han_is_coll_dynamic_implemented(coll)) {
            continue;
        }
        for(topo_lvl = 0; topo_lvl < NB_TOPO_LVL; topo_lvl++) {

            snprintf(param_name, sizeof(param_name), "%s_dynamic_%s_module",
                     mca_coll_base_colltype_to_str(coll),
                     mca_coll_han_topo_lvl_to_str(topo_lvl));

            param_desc_size = snprintf(param_desc, sizeof(param_desc),
                                       "Collective module to use for %s on %s topological level: ",
                                       mca_coll_base_colltype_to_str(coll),
                                       mca_coll_han_topo_lvl_to_str(topo_lvl));
            /*
             * Exhaustive description:
             * 0 = self; 1 = basic; 2 = libnbc; ...
             * FIXME: Do not print component not providing this collective
             */
            for(component = 0 ; component < COMPONENTS_COUNT ; component++) {
                if(HAN == component && GLOBAL_COMMUNICATOR != topo_lvl) {
                    /* Han can only be used on the global communicator */
                    continue;
                }
                param_desc_size += snprintf(param_desc+param_desc_size, sizeof(param_desc) - param_desc_size,
                                            "%d = %s; ",
                                            component,
                                            ompi_coll_han_available_components[component].component_name);
            }

            mca_base_component_var_register(c, param_name, param_desc,
                                            MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                            OPAL_INFO_LVL_9,
                                            MCA_BASE_VAR_SCOPE_READONLY,
                                            &(cs->mca_sub_components[coll][topo_lvl]));
        }
    }

    /* Split choices */
    for (int split_lvl = 0 ; split_lvl < NB_SPLIT_LVL ; split_lvl++) {
        cs->split_requested[split_lvl] = 0;
    }

    int nlevel;
    char ** supported_splits = NULL;
    for (int split_lvl = 0 ; split_lvl < NB_SPLIT_LVL ; split_lvl++) {
            opal_argv_append(&nlevel, &supported_splits,
                             mca_coll_han_split_lvl_to_str(split_lvl));
    }
    char desc [1024];
    snprintf(desc, sizeof(desc),
             "Comma separated list of topological splits desired (supported values: %s)",
             opal_argv_join(supported_splits, ','));


    cs->splits = malloc(sizeof(char *));
    *cs->splits = NULL;
    mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                    "splits",
                                    desc,
                                    MCA_BASE_VAR_TYPE_STRING, NULL, 0,
                                    0, OPAL_INFO_LVL_6,
                                    MCA_BASE_VAR_SCOPE_READONLY,
                                    cs->splits);
    if (NULL != *cs->splits) {
        char ** splits = opal_argv_split(*cs->splits, ',');
        for (int split_lvl = 0 ; split_lvl < NB_SPLIT_LVL ; split_lvl++) {
            for (char** split = splits; split && *split; ++split) {
                if (!strcmp(mca_coll_han_split_lvl_to_str(split_lvl), *split)) {
                    cs->split_requested[split_lvl] = true;
                }
            }
        }
    } else {
        cs->split_requested[NODE] = 1;
    }

    /* Alltoall algorithm */
    cs->alltoall_algorithm = 2;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "alltoall_algorithm",
                                           "Alltoall algorithm choice. "
                                           "0: grid, 1: pipelined grid, 2: rolling igatherw",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEPRECATED,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->alltoall_algorithm));

    /* Allgather algorithm */
    cs->allgather_algorithm = 3;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "allgather_algorithm",
                                           "Allgather algorithm choice. "
                                           "0: simple, 1: up_comm first, 2: low_comm first, 3: simple_splitted",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0,
                                           MCA_BASE_VAR_FLAG_DEPRECATED,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->allgather_algorithm));

    /* Allgather Simple Splitted algorithm ibcast loop */
    cs->allgather_split_ibcast = false;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "allgather_split_ibcast",
                                           "Loop on ibcast (instead of bcast) in the final phase of "
                                           "Allgather split algorithm, disabled by default",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->allgather_split_ibcast));

    /* Splitting threshold for recursive splitting allreduce algorithm */
    cs->allreduce_min_recursive_split_size = 1024;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "allreduce_min_recursive_split_size",
                                           "Allreduce recursive splitting threshold, "
                                           "usefull with allreduce recursive splitting algorithm",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->allreduce_min_recursive_split_size));

    /* Dynamic rules */
    cs->use_dynamic_file_rules = false;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "use_dynamic_file_rules",
                                           "Enable the dynamic selection provided via the dynamic_rules_filename MCA",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->use_dynamic_file_rules));
    
    /* Pipelined bcast MCA parameters */
    cs->bcast_pipeline_start_size = 4096;
    (void) mca_base_component_var_register(c, "bcast_pipeline_start_size",
                                           "Minimal message size to use pipelined bcast algorithms (default is 4096)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->bcast_pipeline_start_size);
    
    cs->bcast_pipeline_segment_count = 8;
    (void) mca_base_component_var_register(c, "bcast_pipeline_segment_count",
                                           "Maximum segment for pipelined bcast algorithms (default is 8)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->bcast_pipeline_segment_count);

    /* Pipelined alltoallv MCA parameters */
    cs->alltoallv_pipeline_segment_count = 8;
    (void) mca_base_component_var_register(c, "alltoallv_pipeline_segment_count",
                                           "Number of segment for pipelined alltoallv algorithm (default is 8)",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY, &cs->alltoallv_pipeline_segment_count);
    if (0 == cs->alltoallv_pipeline_segment_count) {cs->alltoallv_pipeline_segment_count = 8;}

    cs->dynamic_rules_filename = NULL;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "dynamic_rules_filename",
                                           "Configuration file containing the dynamic selection rules",
                                           MCA_BASE_VAR_TYPE_STRING, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->dynamic_rules_filename));

    cs->dump_dynamic_rules = false;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "dump_dynamic_rules",
                                           "Switch used to decide if we dump  dynamic rules provided by configuration file",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->dump_dynamic_rules));

    if((cs->dump_dynamic_rules || NULL != cs->dynamic_rules_filename)
       && !cs->use_dynamic_file_rules) {
        opal_output_verbose(0, cs->han_output,
                            "HAN: dynamic rules for collectives are hot activated."
                            "Check coll_han_use_dynamic_file_rules MCA parameter");
    }

    cs->max_dynamic_errors = 10;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "max_dynamic_errors",
                                           "Number of dynamic rules module/function "
                                           "errors printed on rank 0 "
                                           "with a 0 verbosity."
                                           "Useless if coll_base_verbose is 30 or more.",
                                           MCA_BASE_VAR_TYPE_INT, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->max_dynamic_errors));

    cs->fake_topo_split = false;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "fake_topo_split",
                                           "Simulate 4 topo level splits. "
                                           "Require that the number of MPI "
                                           "ranks to be a multiple of 8 to be "
                                           "balanced.",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->fake_topo_split));

    cs->balanced_fake_topo_split = true;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "balanced_fake_topo_split",
                                           "Whether the fake topo level splits are balanced or not."
                                           " Requires fake_topo_split=true",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->balanced_fake_topo_split));

    cs->fake_topo_split_by_blocks = true;
    (void) mca_base_component_var_register(&mca_coll_han_component.super.collm_version,
                                           "fake_topo_split_by_blocks",
                                           "Whether the fake topology emulates "
                                           "a linear (a.k.a \"block\" in slurm terminology"
                                           "process distribution across resources"
                                           " Requires fake_topo_split=true"
                                           " To make the topology balanced you need to"
                                           " give a number of processes multiple of 8",
                                           MCA_BASE_VAR_TYPE_BOOL, NULL, 0, 0,
                                           OPAL_INFO_LVL_6,
                                           MCA_BASE_VAR_SCOPE_READONLY,
                                           &(cs->fake_topo_split_by_blocks));


    return OMPI_SUCCESS;
}
