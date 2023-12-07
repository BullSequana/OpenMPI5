/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/**
 * @file
 *
 * Warning: this is not for the faint of heart -- don't even bother
 * reading this source code if you don't have a strong understanding
 * of nested data structures and pointer math (remember that
 * associativity and order of C operations is *critical* in terms of
 * pointer math!).
 */

#include "ompi_config.h"

#include "mpi.h"
#include "coll_han.h"
#include "coll_han_dynamic.h"
#include "ompi/proc/proc.h"
#include <ctype.h>
#include "opal/mca/hwloc/base/base.h"

#define HAN_SUBCOM_SAVE_COLLECTIVE(FALLBACKS, COMM, HANM, COLL)                  \
    do {                                                                         \
        (FALLBACKS).COLL.module_fn.COLL = (COMM)->c_coll->coll_ ## COLL;         \
        (FALLBACKS).COLL.module = (COMM)->c_coll->coll_ ## COLL ## _module;      \
        (COMM)->c_coll->coll_ ## COLL = (HANM)->fallback.COLL.module_fn.COLL;    \
        (COMM)->c_coll->coll_ ## COLL ## _module = (HANM)->fallback.COLL.module; \
    } while(0)

#define HAN_SUBCOM_LOAD_COLLECTIVE(FALLBACKS, COMM, HANM, COLL)                  \
    do {                                                                         \
        (COMM)->c_coll->coll_ ## COLL = (FALLBACKS).COLL.module_fn.COLL;         \
        (COMM)->c_coll->coll_ ## COLL ## _module = (FALLBACKS).COLL.module;      \
    } while(0)


typedef struct {
    int level[NB_SPLIT_LVL];
} split_colors_t;

static int
get_split_type(SPLIT_LVL_T split_lvl)
{
    switch (split_lvl) {
        case SOCKET:
            return OMPI_COMM_TYPE_SOCKET;
        case NODE:
            return MPI_COMM_TYPE_SHARED;
        case CLUSTER:
            return OMPI_COMM_TYPE_CLUSTER;
        default:
            return MPI_UNDEFINED;
    }
}

/* Based ompi/comm/comm.c: ompi_comm_split_type_get_part
 * Use the smallest rank on our part as id
 */
static int
get_my_split_part_id(ompi_group_t *group, const int split_type)
{
    int ret;
    int id = -1;
    int size = ompi_group_size(group);

    for (int rank = 0 ; rank < size ; ++rank) {
        ompi_proc_t *proc = ompi_group_get_proc_ptr_raw (group, rank);
        uint16_t locality;
        uint16_t *u16ptr;
        int include = false;

        if (ompi_proc_is_sentinel (proc)) {
            opal_process_name_t proc_name = ompi_proc_sentinel_to_name ((uintptr_t) proc);

            if (split_type <= OMPI_COMM_TYPE_HOST) {
                /* local ranks should never be represented by sentinel procs. ideally we
                 * should be able to use OPAL_MODEX_RECV_VALUE_OPTIONAL but it does have
                 * some overhead. update this to use the optional recv if that is ever fixed. */
                continue;
            }

            u16ptr = &locality;

            OPAL_MODEX_RECV_VALUE_OPTIONAL(ret, PMIX_LOCALITY, &proc_name, &u16ptr, PMIX_UINT16);
            if (OPAL_SUCCESS != ret) {
                continue;
            }
        } else {
            locality = proc->super.proc_flags;
        }

        switch (split_type) {
            case OMPI_COMM_TYPE_HWTHREAD:
                include = OPAL_PROC_ON_LOCAL_HWTHREAD(locality);
                break;
            case OMPI_COMM_TYPE_CORE:
                include = OPAL_PROC_ON_LOCAL_CORE(locality);
                break;
            case OMPI_COMM_TYPE_L1CACHE:
                include = OPAL_PROC_ON_LOCAL_L1CACHE(locality);
                break;
            case OMPI_COMM_TYPE_L2CACHE:
                include = OPAL_PROC_ON_LOCAL_L2CACHE(locality);
                break;
            case OMPI_COMM_TYPE_L3CACHE:
                include = OPAL_PROC_ON_LOCAL_L3CACHE(locality);
                break;
            case OMPI_COMM_TYPE_SOCKET:
                include = OPAL_PROC_ON_LOCAL_SOCKET(locality);
                break;
            case OMPI_COMM_TYPE_NUMA:
                include = OPAL_PROC_ON_LOCAL_NUMA(locality);
                break;
            case MPI_COMM_TYPE_SHARED:
                include = OPAL_PROC_ON_LOCAL_NODE(locality);
                break;
            case OMPI_COMM_TYPE_BOARD:
                include = OPAL_PROC_ON_LOCAL_BOARD(locality);
                break;
            case OMPI_COMM_TYPE_HOST:
                include = OPAL_PROC_ON_LOCAL_HOST(locality);
                break;
            case OMPI_COMM_TYPE_CU:
                include = OPAL_PROC_ON_LOCAL_CU(locality);
                break;
            case OMPI_COMM_TYPE_CLUSTER:
                include = OPAL_PROC_ON_LOCAL_CLUSTER(locality);
                break;
            default:
                include = 0;
                break;
        }

        if (include && (rank < id || -1 == id)) {
            id = rank;
        }
    }

    return id;
}

static void
get_fake_locality_infos(struct ompi_communicator_t *comm,
                        split_colors_t *my_colors)
{
    int rank = ompi_comm_rank(comm);
    if (mca_coll_han_component.fake_topo_split_by_blocks) {
        int split_rank = ompi_comm_size(comm)/2;
        for (int split_lvl = NB_SPLIT_LVL-1; split_lvl >= 0; split_lvl--) {
            my_colors->level[split_lvl] = (rank >= split_rank);
            rank %= split_rank;
            split_rank /= 2;
        }
    } else if (mca_coll_han_component.balanced_fake_topo_split) {
        for (int split_lvl = 0; split_lvl < NB_SPLIT_LVL; split_lvl++) {
            my_colors->level[split_lvl] = rank%2;
            rank /=2;
        }
    } else {
        switch (rank) {
            case 0:
            case 3:
                my_colors->level[SOCKET]  = 0;
                my_colors->level[NODE]    = 0;
                my_colors->level[CLUSTER] = 0;
                break;
            case 1:
            case 5:
                my_colors->level[SOCKET]  = 1;
                my_colors->level[NODE]    = 0;
                my_colors->level[CLUSTER] = 0;
                break;
            case 2:
            case 8:
                my_colors->level[SOCKET]  = 2;
                my_colors->level[NODE]    = 2;
                my_colors->level[CLUSTER] = 2;
                break;
            case 4:
                my_colors->level[SOCKET]  = 4;
                my_colors->level[NODE]    = 4;
                my_colors->level[CLUSTER] = 4;
                break;
            case 6:
            case 7:
            case 9:
                my_colors->level[SOCKET]  = 6;
                my_colors->level[NODE]    = 2;
                my_colors->level[CLUSTER] = 2;
                break;
            default:
                for (int split_lvl = 0; split_lvl < NB_SPLIT_LVL; split_lvl++) {
                    my_colors->level[split_lvl] = 11;
                }
                break;
        }
    }
}

static void
get_my_locality_infos(struct ompi_communicator_t *comm, split_colors_t *my_colors)
{
    for (int split_lvl = 0 ; split_lvl < NB_SPLIT_LVL ; split_lvl++) {
        if (mca_coll_han_component.split_requested[split_lvl]) {
            int split_type = get_split_type(split_lvl);

            if (MPI_UNDEFINED != split_type) {
                my_colors->level[split_lvl] = get_my_split_part_id(comm->c_local_group, split_type);
            } else {
                /* TODO: something clever if we need to cover split levels
                 *       unsupported by comm_split_type
                 */
                my_colors->level[split_lvl] = 0;
            }
        } else {
            /* This split level is not requested
             * <=> everyone is in the same part
             */
            my_colors->level[split_lvl] = 0;
        }
    }
}

typedef struct topo_s topo_t;
struct topo_s {
    /* Number of diffent socket/node/cluster color for a topologic level*/
    int sub_group_count;

    /* 2 Arrays of size sub_group_count:
     * sub_group_color[i] = socket/node/cluster color used if a MPI_comm_split was called
     * sub_group_size[i] = number of rank with this color */
    int *sub_group_color;
    int *sub_group_size;

    /* Which Split level this topo_t store */
    int current_level;

    /* This 2d array split each rank according to their colors
     * Example at split socket level:
     * 4 Ranks with ranks 0, 1 on socket_0 and ranks 2, 3 on socket_1
     * We have 2 differents sockets so sub_group_count = 2
     * sub_group_color[0] = socket_0
     * sub_group_color[1] = socket_1
     * We have 2 ranks on each socket
     * sub_group_size[0] = 2
     * sub_group_size[1] = 2
     * so group_rank size will be : group_rank[sub_group_count][sub_group_size[i]]
     * it will be fill like thats :
     * for socket 0 => group_rank[0][0] = rank 0
     *              => group_rank[0][1] = rank 1
     * for socket 1 => group_rank[1][0] = rank 2
     *              => group_rank[1][1] = rank 3
     */
    int **group_rank;

    /* Index of my rank in group_rank
     * Same Example than above :
     * I'm rank 2
     * my_sub_group= 1
     * my_sub_group_index = 0
     * (with group_rank[i][j] => my_sub_group = i
     *                        => my_sub_group_index = j*/
    int my_sub_group;
    int my_sub_group_index;

    /* Next split level, one structure is created per color at current level*/
    topo_t *next;
};

/* This function fill sub_group variables in topo struct */
static void
analyse_topo_lvl(const split_colors_t *split_colors,
                 const int *ranks,
                 int ranks_size,
                 SPLIT_LVL_T split_lvl,
                 topo_t *t)
{
    int colors_count = ranks_size;
    /* First we need to know how many differents values
     * there are for a specific topologic level
     *
     * If another rank have the same sock color, remove a drawer
     */
    for (int rank = 0 ; rank < ranks_size ; rank++) {
        for (int successor = rank+1 ; successor < ranks_size ; successor++) {
            int sub_group_rank = ranks[rank];
            int sub_group_next_rank = ranks[successor];
            int rank_color = split_colors[sub_group_rank].level[split_lvl];
            int next_rank_color = split_colors[sub_group_next_rank].level[split_lvl];
            if (rank_color == next_rank_color) {
                colors_count--;
                break;
            }
        }
    }

    /* Now we know how many arrays of ranks we have to store */
    t->sub_group_count = colors_count;

    /* Then we fill sub_group arrays:
     * sub_group_color[i] = Cluster/Node/Socket color
     * sub_group_size[i] = Numbers of ranks which share the same color on a specific split level */
    t->sub_group_color = malloc(colors_count * sizeof(int));
    t->sub_group_size = malloc(colors_count * sizeof(int));
    for (int color = 0 ; color < colors_count ; color++) {
        t->sub_group_color[color] = -1;
        t->sub_group_size[color] =  0;
    }

    for (int rank = 0 ; rank < ranks_size ; rank++) {
        for (int color = 0 ; color < colors_count ; color++) {
            int sub_group_rank = ranks[rank];
            int rank_color = split_colors[sub_group_rank].level[split_lvl];
            if(rank_color != t->sub_group_color[color]) {
                if(t->sub_group_color[color] == -1) {
                    /* We are the first rank with this color */
                    t->sub_group_color[color]   = rank_color;
                    t->sub_group_size[color] = 1;
                    break;
                }
            } else {
                /* Another rank already have this value */
                t->sub_group_size[color]++;
                break;
            }
        }
    }
}

/* Fill all topo struct */
static void
fill_topo(split_colors_t *colors, int *ranks, int ranks_size, int my_rank, SPLIT_LVL_T split_lvl, topo_t *t)
{
    t->current_level = split_lvl;

    /* Fill sub_group_count and sub_groups */
    analyse_topo_lvl(colors, ranks, ranks_size, split_lvl, t);

    /* Allocate several arrays to sort ranks in their topology
     * Example: group_rank[i][j] and group_rank[i][j+1] are on the same Cluster/Node/Socket
     * group_rank[i][j] and group_rank[i+1][j] are on different Cluster/Node/Socket */
    t->group_rank =  malloc(t->sub_group_count * sizeof(int*));
    for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
        if (OPAL_LIKELY(t->sub_group_size[sub_group] > 0)) {
            t->group_rank[sub_group] = malloc(t->sub_group_size[sub_group] * sizeof(int));
        }
    }

    t->my_sub_group = -1;
    t->my_sub_group_index = -1;

    /* Fill each arrays */
    for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
        int member = 0;
        for (int rank = 0 ; rank < ranks_size ; rank++) {
            if (OPAL_UNLIKELY(member >= t->sub_group_size[sub_group])) {
                /* This sub_group is full, go to the next group */
                break;
            }
            int sub_group_rank = ranks[rank];
            if (colors[sub_group_rank].level[split_lvl] == t->sub_group_color[sub_group]) {
                if (ranks[rank] == my_rank) {
                    t->my_sub_group = sub_group;
                    t->my_sub_group_index = member;
                }
                t->group_rank[sub_group][member] = ranks[rank];
                member++;
            }
        }
    }

    if (0 >= split_lvl) {
        /* We are on the bottom of the topology tree */
        t->next = NULL;
        return;
    } else {
        /* Recursive call on each sub_group */
        t->next = malloc(t->sub_group_count * sizeof(topo_t));
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            fill_topo(colors,
                      t->group_rank[sub_group],
                      t->sub_group_size[sub_group],
                      my_rank,
                      split_lvl-1,
                      &t->next[sub_group]);
        }
    }
    return;
}

/* Follow in the topo struct a rank,
 * when the correct split_lvl is reach topo is returned,
 * else NULL is return */
static topo_t*
find_my_topo(topo_t *t, SPLIT_LVL_T split_lvl, int my_rank)
{
    if (my_rank != t->group_rank[t->my_sub_group][t->my_sub_group_index]) {
        /* My rank isn't in this topology level*/
        return NULL;
    }
    if (t->current_level == split_lvl) {
        return t;
    } else {
        return find_my_topo(&t->next[t->my_sub_group], split_lvl, my_rank);
    }
}

/* Fill sub_ranks array in han module from topo
 * Each ranks have to read the entire topo to fill sub_ranks*/
static void
fill_sub_rank(topo_t *t, TOPO_LVL_T topo_lvl, mca_coll_han_module_t *han_module)
{
    if (0 < t->current_level) {
        /* Iner topology nodes */

        /* Fill sub_rank information we can gather here */
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            for (int member = 0 ; member < t->sub_group_size[sub_group] ; member++) {
                han_module->sub_ranks[t->group_rank[sub_group][member] * (NB_TOPO_LVL-1) + topo_lvl] = sub_group;
            }
        }

        /* Recursive call on sub_groups */
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            fill_sub_rank(&t->next[sub_group], topo_lvl-1, han_module);
        }
    } else {
        /* Leaf topology nodes */

        /* Fill current topo_lvl + leaf_lvl information */
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            for (int member = 0 ; member < t->sub_group_size[sub_group] ; member++) {
                han_module->sub_ranks[t->group_rank[sub_group][member] * (NB_TOPO_LVL-1) + topo_lvl] = sub_group;
                han_module->sub_ranks[t->group_rank[sub_group][member] * (NB_TOPO_LVL-1) + topo_lvl-1] = member;
            }
        }
    }
}

/* Find my peer on peer_topo
 * Dive into my_topo and peer_topo symmetrically
 *
 * my_topo:   part of the topo I am in
 * peer_topo: part of the topo I am searching a peer in
 * common_sub_group_id: which sub_group of my_topo I am in
 *                      = which sub_group of peer_topo I am searching a peer in
 *
 * Returns peer rank if there is one, otherwise returns -1
 */
static int
find_my_peer(topo_t *my_topo, topo_t *peer_topo, int common_sub_group_id)
{
    if (peer_topo->sub_group_count <= common_sub_group_id) {
        /* No peer on this peer_topo can match my sub_ranks */
        return -1;
    }

    if (0 == my_topo->current_level) {
        /* We are at the bottom of the topology */
        int my_sub_group_index = my_topo->my_sub_group_index;

        if (peer_topo->sub_group_size[common_sub_group_id] <= my_sub_group_index) {
            /* Not enough ranks on this part of this topoly to find a peer */
            return -1;
        } else {
            /* I found a peer on this part of the topology */
            return peer_topo->group_rank[common_sub_group_id][my_sub_group_index];
        }
    } else {
        /* Go deeper */
        int my_sub_group_id = my_topo->my_sub_group;
        int my_next_sub_group_id = my_topo->next[my_sub_group_id].my_sub_group;

        return find_my_peer(&( my_topo->next[my_sub_group_id] ),
                            &( peer_topo->next[common_sub_group_id] ),
                            my_next_sub_group_id);
    }
}

/* Find my peers on this topo and store them in the ranks table
 * My peers are ranks in this topo which are placed in the same
 * place as me in the topo sub_groups
 *
 * The sized returned are the actual number of peers
 * This size can be smaller than t->nb_tab
 */
static int
find_my_peers(topo_t *t, int *ranks, int w_rank)
{
    int size = 0;
    int peer;
    /* Search one peer for each sub_group bellow this level */
    for (int sub_group=0 ; sub_group < t->sub_group_count ; sub_group++) {
        if(sub_group == t->my_sub_group) {
            peer = w_rank;
        } else {
            peer = find_my_peer(t, t, sub_group);
        }

        /* If there is a peer, add it */
        if (peer >= 0) {
            ranks[size] = peer;
            size++;
        }
    }

    /* Size is at least 1 because I will always find myself in the topo */
    return size;
}

/* Computes sub-communicator rank -> global rank translation table
 * from global rank-> sub-communicator rank translation table
 *
 * This table represents a carthesian grid for global communicators
 * Empty places are set to -1 in case of sub-communicator repartition imbalance*/

static void
compute_translation_table_multi_lvl(struct ompi_communicator_t *comm,
                                    mca_coll_han_module_t *han_module)
{
    int w_rank;
    int sub_rank;
    int w_size = ompi_comm_size(comm);
    int topo_lvl;
    int translation_table_size;
    int i;

    /* Compute maximum sub-communicator sizes */
    for (w_rank = 0 ; w_rank < w_size ; w_rank++) {
        for (topo_lvl = 0 ; topo_lvl < NB_TOPO_LVL-1; topo_lvl++) {
            sub_rank = mca_coll_han_get_sub_ranks(han_module, w_rank)[topo_lvl];
            if (han_module->maximum_size[topo_lvl] < sub_rank+1) {
                han_module->maximum_size[topo_lvl] = sub_rank+1;
            }
        }
    }

    /* Compute translation table size */
    translation_table_size = 1;
    for (topo_lvl = 0 ; topo_lvl < NB_TOPO_LVL-1; topo_lvl++) {
        translation_table_size *= han_module->maximum_size[topo_lvl];
    }

    /*
     * Allocate and initialize translation table
     * Use -1 as an invalid rank
     */
    han_module->global_ranks = (int *) malloc(sizeof(int)*translation_table_size);
    for (i = 0 ; i < translation_table_size ; i++) {
        han_module->global_ranks[i] = -1;
    }

    /*
     * Fill in non empty places in translation table
     *
     * Example for 4 topologics level:
     * LL = leaf_level
     * S = inter_socket
     * N = inter node
     * C = inter cluster
     * ==> global_ranks[rank[C]  * size[LL]
     *                           * size[S]
     *                           * size[N]
     *                 + rank[N] * size[LL]
     *                           * size[S]
     *                 + rank[S] * size[LL]
     *                 + rank[LL]] = w_rank
     */
    for (w_rank = 0 ; w_rank < w_size ; w_rank ++) {
        int translation_table_place = 0;
        for(topo_lvl = GLOBAL_COMMUNICATOR-1; topo_lvl >= 0; topo_lvl--) {
            translation_table_place = translation_table_place * han_module->maximum_size[topo_lvl] +
                                      mca_coll_han_get_sub_ranks(han_module, w_rank)[topo_lvl];
        }
        if (translation_table_place != w_rank) {
            han_module->is_mapbycore = false;
        }
        han_module->global_ranks[translation_table_place] = w_rank;
    }

    /* If a -1 have not been overwritten,
     * there is a hole somewhere in ranks representation
     * so topology is somewhere imbalanced.
     */
    han_module->are_ppn_imbalanced = false;
    for (i = 0 ; i < translation_table_size ; i++) {
        if (-1 == han_module->global_ranks[i]) {
            han_module->are_ppn_imbalanced = true;
            break;
        }
    }
}

static void
compute_rank_from_topo_index_table(topo_t *t, int *rank_from_topo_index)
{
    int n_added = 0;
    /* DFS to compute translation table */

    if (0 >= t->current_level) {
        /* Last split level, simply copy ranks */
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            memcpy (rank_from_topo_index+n_added, t->group_rank[sub_group], t->sub_group_size[sub_group]*sizeof(int));
            n_added += t->sub_group_size[sub_group];
        }
    } else {
        /* Recursibely call on each sub_tree */
        for (int sub_group = 0 ; sub_group < t->sub_group_count ; sub_group++) {
            compute_rank_from_topo_index_table(&t->next[sub_group], rank_from_topo_index + n_added);
            n_added += t->sub_group_size[sub_group];
        }
    }
}

static void
compute_topo_index_from_rank_table(const int *rank_from_topo_index, int *topo_index_from_rank, int size)
{
    for (int rank = 0 ; rank < size ; rank++) {
        topo_index_from_rank[rank_from_topo_index[rank]] = rank;
    }
}

static void
compute_sub_tree(topo_t *t, mca_coll_han_topo_tree_node_t *node)
{
    int n_added = 0;
    node->n_sub_tree = t->sub_group_count;
    node->sub_tree = malloc(node->n_sub_tree * sizeof(mca_coll_han_topo_tree_node_t));

    for (int child = 0 ; child < node->n_sub_tree ; child++) {
        /* Alias */
        mca_coll_han_topo_tree_node_t *sub_tree = &node->sub_tree[child];

        sub_tree->root_rank = t->group_rank[child][0];

        sub_tree->wranks_range.start = node->wranks_range.start + n_added;
        sub_tree->wranks_range.nb = t->sub_group_size[child];
        n_added += sub_tree->wranks_range.nb;

        if (0 == t->current_level) {
            sub_tree->n_sub_tree = 0;
            sub_tree->sub_tree = NULL;
        } else {
            compute_sub_tree(&t->next[child], sub_tree);
        }
    }
}

static int
dump_tree(mca_coll_han_topo_tree_t *topo_tree, char *str, int n_free,
          mca_coll_han_topo_tree_node_t *sub_tree, int level)
{
    int n_written = 0;
    for (int i = 0 ; i < level ; i++) {
        n_written += snprintf(str + n_written, n_free - n_written, "\t");
    }
    n_written += snprintf(str + n_written, n_free - n_written,
                          "root %d, n_sub_tree %d, nranks %d, ranks ( ",
                          sub_tree->root_rank, sub_tree->n_sub_tree, sub_tree->wranks_range.nb);

    for (int i = 0 ; i < sub_tree->wranks_range.nb ; i++) {
        n_written += snprintf(str + n_written, n_free - n_written,
                              "%d ", topo_tree->rank_from_topo_index[sub_tree->wranks_range.start + i]);
    }
    n_written += snprintf(str + n_written, n_free - n_written, ")\n");

    for (int i = 0 ; i < sub_tree->n_sub_tree ; i++) {
        n_written += dump_tree(topo_tree, str + n_written, n_free - n_written,
                               &sub_tree->sub_tree[i], level+1);
    }

    return n_written;
}

static void
find_my_sub_tree(mca_coll_han_topo_tree_t *topo_tree, int my_id)
{
    int left;
    int middle;
    int right;
    mca_coll_han_topo_tree_node_t *sub_tree = topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1];

    for (int topo_lvl = GLOBAL_COMMUNICATOR-2 ; topo_lvl >= LEAF_LEVEL ; topo_lvl--) {
        left = 0;
        right = sub_tree->n_sub_tree-1;
        while (left+1 < right) {
            middle = (left + right) / 2;
            if (my_id >= sub_tree->sub_tree[middle].wranks_range.start) {
                left = middle;
            } else {
                right = middle;
            }
        }
        if (my_id >= sub_tree->sub_tree[right].wranks_range.start) {
            middle = right;
        } else {
            middle = left;
        }

        topo_tree->my_sub_tree[topo_lvl] = &sub_tree->sub_tree[middle];
        sub_tree = &sub_tree->sub_tree[middle];
    }
}

static void
compute_topo_tree(topo_t *t, struct ompi_communicator_t *comm, mca_coll_han_module_t *han_module)
{
    int comm_size = ompi_comm_size(comm);
    mca_coll_han_topo_tree_t *topo_tree = malloc(sizeof(mca_coll_han_topo_tree_t));

    topo_tree->rank_from_topo_index = malloc(comm_size * sizeof(int));
    topo_tree->topo_index_from_rank = malloc(comm_size * sizeof(int));

    compute_rank_from_topo_index_table(t, topo_tree->rank_from_topo_index);
    compute_topo_index_from_rank_table(topo_tree->rank_from_topo_index, topo_tree->topo_index_from_rank, comm_size);

    topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1] = malloc(sizeof(mca_coll_han_topo_tree_node_t));
    topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]->root_rank = 0;
    topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]->wranks_range.start = 0;
    topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]->wranks_range.nb = comm_size;
    compute_sub_tree(t, topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]);

    find_my_sub_tree(topo_tree, topo_tree->topo_index_from_rank[ompi_comm_rank(comm)]);

    /* Dump topo tree  */
    if (0 == ompi_comm_rank(comm) && 0 < mca_coll_han_component.han_output_verbose) {
        char topo_tree_str[65536];
        topo_tree_str[0] = '\0';
        int n_free = 65536;
        int n_written = 0;
        n_written += snprintf(topo_tree_str+n_written, n_free - n_written,
                              "rank_from_topo_index: ");

        for (int i = 0 ; i < comm_size ; i++) {
            n_written += snprintf(topo_tree_str+n_written, n_free - n_written,
                                  "%d ", topo_tree->rank_from_topo_index[i]);
        }
        n_written += snprintf(topo_tree_str + n_written, n_free - n_written, "\n");

        n_written += snprintf(topo_tree_str+n_written, n_free - n_written,
                              "topo_index_from_rank: ");
        for (int i = 0 ; i < comm_size ; i++) {
            n_written += snprintf(topo_tree_str+n_written, n_free - n_written,
                                  "%d ", topo_tree->topo_index_from_rank[i]);
        }
        n_written += snprintf(topo_tree_str + n_written, n_free - n_written, "\n");

        n_written += dump_tree(topo_tree, topo_tree_str+n_written, n_free - n_written,
                               topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1], 0);

        opal_output(mca_coll_han_component.han_output,
                    "coll::han::mca_coll_han_comm_create_multi_level"
                    "::dump_topo_tree for communicator (%d/%s)\n"
                    "%s",
                    comm->c_contextid, comm->c_name,
                    topo_tree_str);
    }

    han_module->topo_tree = topo_tree;
}

static void
smash_topo_tree_node(mca_coll_han_topo_tree_node_t *topo_tree_node)
{
    for (int child = 0; child < topo_tree_node->n_sub_tree; child++) {
        smash_topo_tree_node(&topo_tree_node->sub_tree[child]);
    }

    free(topo_tree_node->sub_tree);
    topo_tree_node->sub_tree = NULL;
}

void
mca_coll_han_smash_topo_tree(mca_coll_han_module_t *han_module)
{
    smash_topo_tree_node(han_module->topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]);
    free(han_module->topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1]);

    for (int topo_lvl = LEAF_LEVEL; topo_lvl < GLOBAL_COMMUNICATOR; topo_lvl++) {
        han_module->topo_tree->my_sub_tree[topo_lvl] = NULL;
    }

    free(han_module->topo_tree->rank_from_topo_index);
    han_module->topo_tree->rank_from_topo_index = NULL;

    free(han_module->topo_tree->topo_index_from_rank);
    han_module->topo_tree->topo_index_from_rank = NULL;

    free(han_module->topo_tree);
    han_module->topo_tree = NULL;
}

static void
free_topo(topo_t *t)
{
    for (int sub_group_id = 0; sub_group_id < t->sub_group_count; sub_group_id++) {
        if (t->current_level > 0) {
            free_topo(&t->next[sub_group_id]);
        }

        free(t->group_rank[sub_group_id]);
        t->group_rank[sub_group_id] = NULL;
    }

    free(t->group_rank);
    t->group_rank = NULL;

    free(t->sub_group_color);
    t->sub_group_color = NULL;

    free(t->sub_group_size);
    t->sub_group_size = NULL;

    free(t->next);
    t->next = NULL;
}

/* Create all differents levels of communicator indicate by MCA parameters */
int mca_coll_han_comm_create_multi_level(struct ompi_communicator_t *comm,
                                         mca_coll_han_module_t *han_module)
{
    int w_size;
    int w_rank;
    mca_coll_han_collectives_fallback_t fallbacks;
    int rc = OMPI_SUCCESS;
    opal_info_t comm_info;
    split_colors_t *split_colors;
    split_colors_t my_colors;

    /* Do this only once */
    if (NULL != han_module->sub_comm[0]) {
        return OMPI_SUCCESS;
    }

    /*
     * We cannot use han allreduce and allgather without sub-communicators,
     * but we are in the creation of the data structures for the HAN, and
     * temporarily need to save back the old collective.
     */
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allgather);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allreduce);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, alltoall);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, alltoallv);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, bcast);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, reduce);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, gather);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, scatter);

    /**
     * HAN is not yet optimized for a single process per node case, we should
     * avoid selecting it for collective communication support in such cases.
     * However, in order to decide if this is true, we need to know how many
     * local processes are on each node, a condition that cannot be verified
     * outside the MPI support (with PRRTE the info will be eventually available,
     * but we don't want to delay anything until then). We can achieve the same
     * goal by using a reduction over the maximum number of peers per node among
     * all participants.
     */
    int local_procs = ompi_group_count_local_peers(comm->c_local_group);
    rc = comm->c_coll->coll_allreduce(MPI_IN_PLACE, &local_procs, 1, MPI_INT,
                                      MPI_MAX, comm,
                                      comm->c_coll->coll_allreduce_module);
    if( OMPI_SUCCESS != rc ) {
        goto return_with_error;
    }
    if( local_procs == 1 ) {
        /* restore saved collectives */
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgather);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allreduce);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoall);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoallv);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, bcast);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, reduce);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, gather);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, scatter);
        han_module->enabled = false;  /* entire module set to pass-through from now on */
        return OMPI_ERR_NOT_SUPPORTED;
    }

    OBJ_CONSTRUCT(&comm_info, opal_info_t);

    /* Create topological sub-communicators */
    for (int topo_lvl = 0 ; topo_lvl < NB_TOPO_LVL-1 ; topo_lvl++) {
        han_module->sub_comm[topo_lvl] = MPI_COMM_SELF;
    }
    w_size = ompi_comm_size(comm);
    w_rank = ompi_comm_rank(comm);

    split_colors = malloc(w_size * sizeof(split_colors_t));
    if (mca_coll_han_component.fake_topo_split) {
        get_fake_locality_infos(comm, &my_colors);
    } else {
        get_my_locality_infos(comm, &my_colors);
    }

    rc = comm->c_coll->coll_allgather(&my_colors,
                                      sizeof(split_colors_t),
                                      MPI_BYTE,
                                      split_colors,
                                      sizeof(split_colors_t),
                                      MPI_BYTE,
                                      comm,
                                      comm->c_coll->coll_allgather_module);
    if( OMPI_SUCCESS != rc ) {
        goto return_with_error;
    }

    /* Fill comm_world ranks tab */
    int *ranks_comm_world = malloc(sizeof(int) * w_size);
    for (int rank=0; rank<w_size; rank++) {
        ranks_comm_world[rank] = rank;
    }

    topo_t global_topo;

    /* Create full topology */
    fill_topo(split_colors, ranks_comm_world, w_size, w_rank, NB_SPLIT_LVL-1, &global_topo);
    free(ranks_comm_world);

    /* Ranks used to create sub_communicators
     * Sub_communicators are smaller than the global communicator
     */
    int *ranks = malloc(sizeof(int) * w_size);

    /* Create Group and communicators from topology */
    for (int split_lvl = 0 ; split_lvl < NB_SPLIT_LVL ; split_lvl++) {

        /* Get topology sub_tree for this level */
        topo_t *sub_topo = find_my_topo(&global_topo, split_lvl, w_rank);

        /* If we are at the bottom of the topology tree,
         * build leaf level as well
         */
        if (0 == split_lvl) {
            int size = sub_topo->sub_group_size[sub_topo->my_sub_group];

            /* Keep MPI_COMM_SELF for this level if I am alone in this group */
            if ( size > 1) {
                ompi_group_t *leaf_group;
                ompi_communicator_t *leaf_comm;

                /* Peers on leaf level are ranks sharing this sub_group with me */
                rc = ompi_group_incl(comm->c_local_group, size, sub_topo->group_rank[sub_topo->my_sub_group], &leaf_group);
                if( OMPI_SUCCESS != rc ) {
                    goto return_with_error;
                }

                /* Let the new communicator know his level
                 * This value should be read by mca_coll_han_comm_query
                 * during the collective component election
                 * for this new communicator
                 */
                opal_info_set(&comm_info, "ompi_comm_coll_han_topo_level", mca_coll_han_topo_lvl_to_str(split_lvl));

                /* Create sub_communicator from group and store it */
                rc = ompi_comm_create_group_with_info(comm, leaf_group, 0, &comm_info, &leaf_comm);
                if( OMPI_SUCCESS != rc ) {
                    goto return_with_error;
                }
                han_module->sub_comm[split_lvl] = leaf_comm;
            }
        }
        int size = sub_topo->sub_group_count;

        /* Keep MPI_COMM_SELF for this level if I am alone in this group */
        if (size > 1) {
            ompi_group_t *split_group;
            ompi_communicator_t *split_comm;

            /* Find my peers and create a group with them */
            size = find_my_peers(sub_topo, ranks, w_rank);
            rc = ompi_group_incl(comm->c_local_group, size, ranks, &split_group);
            if( OMPI_SUCCESS != rc ) {
                goto return_with_error;
            }

            /* Let the new communicator know his level
             * This value should be read by mca_coll_han_comm_query
             * during the collective component election
             * for this new communicator
             */
            opal_info_set(&comm_info, "ompi_comm_coll_han_topo_level", mca_coll_han_topo_lvl_to_str(split_lvl+1));

            /* Create sub_communicator from group and store it */
            rc = ompi_comm_create_group_with_info(comm, split_group, 0, &comm_info, &split_comm);
            if( OMPI_SUCCESS != rc ) {
                goto return_with_error;
            }
            han_module->sub_comm[split_lvl+1] = split_comm;
        }
    }
    free(ranks);

    /* Fill sub_ranks and translation table */
    han_module->sub_ranks = malloc(sizeof(int)*(NB_TOPO_LVL-1)*w_size);
    fill_sub_rank(&global_topo, GLOBAL_COMMUNICATOR-1, han_module);

    compute_translation_table_multi_lvl(comm, han_module);

    /* Store compacted topo tree */
    compute_topo_tree(&global_topo, comm, han_module);

    /* Do not need topo structure anymore */
    free_topo(&global_topo);

    /* Reset the saved collectives to point back to HAN */
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgather);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allreduce);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoall);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoallv);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, bcast);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, reduce);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, gather);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, scatter);
    OBJ_DESTRUCT(&comm_info);

    return OMPI_SUCCESS;

return_with_error:
    return rc;
}

/* Return a tab of subranks from the world rank */
const int*
mca_coll_han_get_sub_ranks(const mca_coll_han_module_t *han_module, int w_rank)
{
    return &(han_module->sub_ranks[w_rank * (NB_TOPO_LVL-1)]);
}

/* Return the world rank from a tab of these sub ranks
 * sub_ranks size must be NB_TOPO_LVL-1
 */
int mca_coll_han_get_global_rank(const mca_coll_han_module_t *han_module,
                                 const int *sub_ranks)
{
    int index = 0;
    for(int topo_lvl = GLOBAL_COMMUNICATOR-1; topo_lvl >= 0; topo_lvl--) {
        index = index * han_module->maximum_size[topo_lvl] +
                sub_ranks[topo_lvl];
    }
    return han_module->global_ranks[index];
}

/**
 * Routine that creates the local hierarchical sub-communicators
 * Called each time a collective is called.
 * comm: input communicator of the collective
 */
int mca_coll_han_comm_create_new(struct ompi_communicator_t *comm,
                                 mca_coll_han_module_t *han_module)
{
    int res = mca_coll_han_comm_create_multi_level(comm, han_module);
    if (res != OMPI_SUCCESS) {
        return res;
    }

    /* Compute vranks for compatibility */
    if (han_module->cached_vranks == NULL) {
        int *vranks;
        int w_size;
        w_size = ompi_comm_size(comm);
        vranks = (int *) malloc(sizeof(int) * w_size);

        for (int rank=0; rank<w_size; rank++) {
            const int *sub;
            sub = mca_coll_han_get_sub_ranks(han_module, rank);
            vranks[rank] = han_module->maximum_size[LEAF_LEVEL] * sub[INTER_NODE] + sub[LEAF_LEVEL];
        }
        han_module->cached_vranks = vranks;
    }
    return res;
}

/*
 * Routine that creates the local hierarchical sub-communicators
 * Called each time a collective is called.
 * comm: input communicator of the collective
 */
int mca_coll_han_comm_create(struct ompi_communicator_t *comm,
                             mca_coll_han_module_t *han_module)
{
    int low_rank, low_size, up_rank, w_rank, w_size;
    mca_coll_han_collectives_fallback_t fallbacks;
    ompi_communicator_t **low_comms;
    ompi_communicator_t **up_comms;
    int vrank, *vranks;
    int err;
    opal_info_t comm_info;

    /* use cached communicators if possible */
    if (han_module->enabled && han_module->cached_low_comms != NULL &&
        han_module->cached_up_comms != NULL &&
        han_module->cached_vranks != NULL) {
        return OMPI_SUCCESS;
    }

    /*
     * We cannot use han allreduce and allgather without sub-communicators,
     * but we are in the creation of the data structures for the HAN, and
     * temporarily need to save back the old collective.
     *
     * Allgather is used to compute vranks
     * Allreduce is used by ompi_comm_split_type in create_intranode_comm_new
     * Reduce + Bcast may be called by the allreduce implementation
     * Gather + Bcast may be called by the allgather implementation
     */
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allgather);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, allreduce);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, alltoall);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, bcast);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, reduce);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, gather);
    HAN_SUBCOM_SAVE_COLLECTIVE(fallbacks, comm, han_module, scatter);

    /**
     * HAN is not yet optimized for a single process per node case, we should
     * avoid selecting it for collective communication support in such cases.
     * However, in order to decide if this is tru, we need to know how many
     * local processes are on each node, a condition that cannot be verified
     * outside the MPI support (with PRRTE the info will be eventually available,
     * but we don't want to delay anything until then). We can achieve the same
     * goal by using a reduction over the maximum number of peers per node among
     * all participants.
     */
    int local_procs = ompi_group_count_local_peers(comm->c_local_group);
    comm->c_coll->coll_allreduce(MPI_IN_PLACE, &local_procs, 1, MPI_INT,
                                 MPI_MAX, comm,
                                 comm->c_coll->coll_allreduce_module);
    if( local_procs == 1 ) {
        /* restore saved collectives */
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgather);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allreduce);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoall);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, bcast);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, reduce);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, gather);
        HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, scatter);
        han_module->enabled = false;  /* entire module set to pass-through from now on */
        return OMPI_ERR_NOT_SUPPORTED;
    }

    /* create communicators if there is no cached communicator */
    w_rank = ompi_comm_rank(comm);
    w_size = ompi_comm_size(comm);
    low_comms = (struct ompi_communicator_t **)malloc(COLL_HAN_LOW_MODULES *
                                                      sizeof(struct ompi_communicator_t *));
    up_comms = (struct ompi_communicator_t **)malloc(COLL_HAN_UP_MODULES *
                                                     sizeof(struct ompi_communicator_t *));

    OBJ_CONSTRUCT(&comm_info, opal_info_t);

    /*
     * Upgrade sm module priority to set up low_comms[0] with sm module
     * This sub-communicator contains the ranks that share my node.
     */
    opal_info_set(&comm_info, "ompi_comm_coll_preference", "tuned,^han");
    err = ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                               &comm_info, &(low_comms[0]));
    if (err != OMPI_SUCCESS) {
        goto failed_lowcomm0;
    }

    /*
     * Get my local rank and the local size
     */
    low_size = ompi_comm_size(low_comms[0]);
    low_rank = ompi_comm_rank(low_comms[0]);

    /*
     * Upgrade shared module priority to set up low_comms[1] with shared module
     * This sub-communicator contains the ranks that share my node.
     */
    opal_info_set(&comm_info, "ompi_comm_coll_preference", "sm,^han");
    err = ompi_comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0,
                         &comm_info, &(low_comms[1]));
    if (err != OMPI_SUCCESS) {
        goto failed_lowcomm1;
    }

    /*
     * Upgrade libnbc module priority to set up up_comms[0] with libnbc module
     * This sub-communicator contains one process per node: processes with the
     * same intra-node rank id share such a sub-communicator
     */
    opal_info_set(&comm_info, "ompi_comm_coll_preference", "libnbc,^han");
    err = ompi_comm_split_with_info(comm, low_rank, w_rank, &comm_info, &(up_comms[0]), false);
    if (err != OMPI_SUCCESS) {
        goto failed_upcomm0;
    }

    up_rank = ompi_comm_rank(up_comms[0]);

    /*
     * Upgrade adapt module priority to set up up_comms[0] with adapt module
     * This sub-communicator contains one process per node.
     */
    opal_info_set(&comm_info, "ompi_comm_coll_preference", "adapt,^han");
    err = ompi_comm_split_with_info(comm, low_rank, w_rank, &comm_info, &(up_comms[1]), false);
    if (err != OMPI_SUCCESS) {
        goto failed_upcomm1;
    }

    /*
     * Set my virtual rank number.
     * my rank # = <intra-node comm size> * <inter-node rank number>
     *             + <intra-node rank number>
     * WARNING: this formula works only if the ranks are perfectly spread over
     *          the nodes
     * TODO: find a better way of doing
     */
    vrank = low_size * up_rank + low_rank;
    vranks = (int *)malloc(sizeof(int) * w_size);
    /*
     * gather vrank from each process so every process will know other processes
     * vrank
     */
    comm->c_coll->coll_allgather(&vrank, 1, MPI_INT, vranks, 1, MPI_INT, comm,
                                 comm->c_coll->coll_allgather_module);

    /*
     * Set the cached info
     */
    han_module->cached_low_comms = low_comms;
    han_module->cached_up_comms = up_comms;
    han_module->cached_vranks = vranks;

    /* Reset the saved collectives to point back to HAN */
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgatherv);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allgather);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, allreduce);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, alltoall);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, bcast);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, reduce);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, gather);
    HAN_SUBCOM_LOAD_COLLECTIVE(fallbacks, comm, han_module, scatter);

    OBJ_DESTRUCT(&comm_info);
    return OMPI_SUCCESS;
    failed_upcomm1:
        ompi_comm_free(&up_comms[0]);
    failed_upcomm0:
        ompi_comm_free(&low_comms[1]);
    failed_lowcomm1:
        ompi_comm_free(&low_comms[0]);
    failed_lowcomm0:
        free(low_comms);
        free(up_comms);
        return err;
}


