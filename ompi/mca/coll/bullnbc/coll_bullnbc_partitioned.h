/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/*
 * @file
 *
 * This file contains the declarations of structures used for partitioned
 * collectives.
 * Partitioned collective communications can either rely on partitioned
 * point to point communications or on a DAG of task wrapping actions
 * including MPI3 communications.
 * Point-to-point based collectives are used for reliability.
 * For example in case local and remote partitions have different sizes.
 * However efficient implementations need more advanced communication
 * patterns including dependencies.
 * Since partitioned collectives cannot start immediately (need for
 * data availability), a DAG of dependency must be stored in the request.
 *
 * In the first case (point-to-points), for each collective communication
 * partition a list stores wich point-to-point partitions need to be
 * completed before completing the collective partition.
 * In this case, a partitioned collectives uses a array of opal_list.
 *
 * In the second case (the DAG), nodes are created and a request has pointer
 * to some of them called entry_nodes. Each node corresponds to a actions.
 * The ready nodes are entry nodes directly connected to the collective partitions.
 * Other entry nodes are ones that must be polled during progress.
 * The progress of those entry nodes may succeed or fail.
 * In case of failure, it will be retried at the next progress call.
 * In case of success, its dependencies are unlock by 1. This dependency
 * is executed if it was the last lock. This makes a pre-order DAG traversal.
 * Entry nodes are MPI_Irecv request completion, Isend, RMA notification wait.
 * Any other nodes must be a dependency of another one. It garantees that
 * it will be executed at some point.
 *
 *
 * Currently both approaches use the same request structure, but with
 * differents function pointers (for progress, start, ready,
 * completion detection and free).
 * The point-to-point partitioned based routines are defined in
 * coll_bullnbc_partitioned.c and DAG routines are located in
 * coll_bullnbc_partitioned_dag.c and coll_bullnbc_partitioned_dag_sched,c
 * files.
 * Requests creations that use these libs are defined in files that
 * defines their MPI conterpart. For example, Palltoall(r) is defined in
 * coll_bullnbc_ialltoall.c
 */

#ifndef MCA_COLL_BULLNBC_PARTITIONED_H
#define MCA_COLL_BULLNBC_PARTITIONED_H

BEGIN_C_DECLS

#include "ompi/mpiext/pcoll_part/c/mpiext_pcoll_part_c.h"
#include "ompi/mca/part/part.h"

#include "ompi/win/win.h"
#include "coll_bullnbc_partitioned_dag.h"

struct ompi_coll_bullnbc_subpart {
    opal_free_list_item_t super;
    ompi_request_t * sub_req;
    int part_idx;
};
typedef struct ompi_coll_bullnbc_subpart ompi_coll_bullnbc_subpart;
OBJ_CLASS_DECLARATION(ompi_coll_bullnbc_subpart);
OMPI_DECLSPEC extern opal_free_list_t subpart_free_list;

/* Partitioned collective request definition.
 * Supports both DAG and partitioned pt2pt engines*/
typedef struct ompi_coll_bullnbc_pcoll_request_t ompi_coll_bullnbc_pcoll_request_t;
struct ompi_coll_bullnbc_pcoll_request_t {
    ompi_request_t req_ompi;

    struct ompi_communicator_t *comm;
    /* Unique tag used by all process in the communicator in send and recv */
    int tag;

    /* Requests registered by user during init to track partition completion */
    size_t total_sparts;
    size_t total_rparts;
    /* For waitnay thread safety, after a request is waited
     * it has its completion status replaced by an invalid value
     * to prevent other threads to capture it again.
     * At collective completion, user subreq may have become invalid, so
     * completions must have been counted before, */
    int32_t n_completed_subreqs;
    ompi_request_t** user_part_reqs;
    ompi_request_t** user_sreqs;
    ompi_request_t** user_rreqs;

    /* Request bound to internal Isend/Irecv */
    int32_t n_internal_reqs;
    int32_t n_internal_reqs_sched;
    int32_t n_internal_reqs_started;
    ompi_request_t** internal_reqs;

    /* ------ data for partitioned pt2pt engine ------- */
    /* arrays of list of parts of pt2pt partitioned,
     * each list contains parts to complete a collective part */
    opal_list_t *send_subparts;
    opal_list_t *recv_subparts;

    /* ------- DAG shared infos ------ */
    int n_started; /* stat for integrity checks */
    int n_complete;
    char * tmpbuf; /* Any malloc'ed buffer to free at the request free */
    ompi_win_t* win; /* Any RMA window to free at the request free */
    int n_created_ddt; /* Any derived datatype to free at the reaquest free */

    ompi_datatype_t** created_ddt;

    opal_list_t active_nodes; /* List of dag items that wait for progress */
    int n_entry_nodes; /* Count of DAG item that have no ancestor */
    int n_ready_nodes; /* Count of DAG item that will be triggered with Pready */
    int n_entry_sched; /* Count of DAG item with no ancestor already scheduled */
    /* All entry points of the DAG : ready + others such as internal recv. */
    pcoll_dag_item ** entry_nodes;

    opal_mutex_t active_list_lock;

    /* To call to make progress*/
    int(*progress_cb)(ompi_coll_bullnbc_pcoll_request_t*req);
};
OBJ_CLASS_DECLARATION(ompi_coll_bullnbc_pcoll_request_t);

/* Free list of request object to describe a entire partitioned-collective */
extern opal_free_list_t mca_coll_bullnbc_pcoll_requests;

/* List  of all active partitioned collective waiting for completion */
extern opal_list_t pending_pcoll_requests;
extern opal_mutex_t pcoll_list_lock;


ompi_coll_bullnbc_pcoll_request_t* ompi_mca_coll_bullnbc_alloc_pcoll_request (void);
int ompi_mca_coll_bullnbc_free (ompi_request_t** request);


/* The request user can wait for each partition */
extern opal_free_list_t part_req_free_list;


int subpart_requests_create( ompi_coll_bullnbc_pcoll_request_t* req);

ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_pcoll_init_empty_req(size_t total_sparts,
                                           ompi_request_t** user_sreqs,
                                           size_t total_rparts,
                                           ompi_request_t** user_rreqs,
                                           struct ompi_communicator_t *comm,
                                           ompi_request_t ** user_main_req);


 /* MCA parameters used by all partitioned collectives */
extern int mca_coll_bullnbc_uniform_rddt;
extern int mca_coll_bullnbc_uniform_partition_sizes;

int bullnbc_partitioned_register(void);
END_C_DECLS

#endif /* MCA_COLL_BULLNBC_PARTITIONED_H */
