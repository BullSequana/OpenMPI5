/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */


/**
 * @file
 *
 * This file provides implementation for partitioned collective requests
 * based on a DAG of point-to-point comunications.
 * This file provides implementation for request creation and free,
 * and for start, pready, parrived and progress functions.
 *
 */

#include "opal/util/show_help.h"

#include "coll_bullnbc.h"
#include "coll_bullnbc_internal.h"
#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_partitioned_dag.h"

#if OMPI_MPI_NOTIFICATIONS
#include "ompi/mpiext/notified_rma/c/mpiext_notifs_rma.h"
#endif /* OMPI_MPI_NOTIFICATIONS */

int mca_coll_bullnbc_uniform_partition_sizes = 0;


int
bullnbc_partitioned_register(void)
{
  mca_base_component_var_register(&mca_coll_bullnbc_component.super.collm_version,
                                  "uniform_partition_count",
                                  "Hint that any partition on the send side contains exactly the data of a single partition on the receive side. Enabled finer send/recv optimizations",
                                  MCA_BASE_VAR_TYPE_BOOL, NULL, 0, MCA_BASE_VAR_FLAG_SETTABLE,
                                  OPAL_INFO_LVL_5,
                                  MCA_BASE_VAR_SCOPE_ALL,
                                  &mca_coll_bullnbc_uniform_partition_sizes);
    return OMPI_SUCCESS;
}
static void unlock(pcoll_dag_item* item);
static int inline do_step(pcoll_dag_item* item);
int ompi_mca_coll_bullnbc_pcoll_update_dag (ompi_coll_bullnbc_pcoll_request_t*req);


#if PROFILE
unsigned long long  profiling [N_DAG_TYPE];
int n_profiled [N_DAG_TYPE];

#define print_profile(type)     \
    if (n_profiled[type])           \
    fprintf(stderr, "%10d runs of %13s took %15llu ns (avg %15lluns)\n",          \
            n_profiled[type], #type, profiling[type], profiling[type]/n_profiled[type]);
#endif

void coll_bullnbc_fini_part_dag(void){
#if PROFILE
    int rank = ompi_comm_rank (&ompi_mpi_comm_world.comm);
    if (rank) return OMPI_SUCCESS;
    print_profile(DAG_CONVERT)
    print_profile(DAG_COPY)
    print_profile(DAG_SEND)
    print_profile(DAG_READY)
    print_profile(DAG_COMPLETE)

fflush(stderr);
#endif
}
#undef print_profile

void coll_bullnbc_init_part_dag(void)
{
#if PROFILE
    int rank = ompi_comm_rank (&ompi_mpi_comm_world.comm);
    if (rank) return;
    for (int i=0; i < N_DAG_TYPE; ++i){
        profiling[i] = 0;
        n_profiled[i] = 0;
    }
#endif
}

/*
 * Called on MPI_Start, this routine set the partitioned collectives as active
 * and starts their internal Irecv */
static int
ompi_mca_coll_bullnbc_dag_start_dag(unsigned long count, ompi_request_t **request)
{

    /* Loop on all partitioned collective request to start */
    for (unsigned long ireq = 0; ireq < count; ++ireq){
        ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request[ireq];

        if (!REQUEST_COMPLETE(&req->req_ompi)){
            NBC_DEBUG(1, "Cannot start coll which is not completed\n");
        }

        /* Set all user sub-requests ready for wait */
        for (size_t i = 0; i < req->total_sparts + req->total_rparts; ++i) {
                ompi_request_t*user_part_req = req->user_part_reqs[i];
                user_part_req->req_state = OMPI_REQUEST_ACTIVE;
                user_part_req->req_complete = REQUEST_PENDING;
        }
        req->n_completed_subreqs = 0;

        /* Set main request ready for wait */
        req->req_ompi.req_state = OMPI_REQUEST_ACTIVE;
        req->req_ompi.req_complete = REQUEST_PENDING;
        req->n_started++;
        req->n_internal_reqs_started = 0;

        /* Start all internal partitioned pt2pt */
        /* register all entry points that do not wait a ready,
         * to make them progress on opal_progress */
        NBC_DEBUG(5, "start coll %p with %d ptpt, %d ready, %d entry \n",
                  req, req->n_internal_reqs,
                  req->n_ready_nodes, req->n_entry_nodes);
        for (int i = req->n_ready_nodes; i < req->n_entry_nodes; ++i) {
            pcoll_dag_item * recv = req->entry_nodes[i];
            NBC_DEBUG(20, "start %p %d/%d\n",
                      recv, i, req->n_entry_nodes);
            if (recv->type == DAG_RECV) {
                MCA_PML_CALL(irecv((void*)recv->args.comm.buf,
                                   recv->args.comm.count,
                                   recv->args.comm.datatype,
                                   recv->args.comm.peer,
                                   req->tag + recv->args.comm.tagshift,
                                   recv->args.comm.comm,
                                   recv->args.comm.req));
                req->n_internal_reqs_started++;
                NBC_DEBUG(5, "start recv %p: addr %p with %d ddt %p from %d, tag %d, "
                          "pml_req %p, collreq %p\n",
                          recv,
                          (void*)recv->args.comm.buf,
                          recv->args.comm.count,
                          recv->args.comm.datatype,
                          recv->args.comm.peer,
                          req->tag + recv->args.comm.tagshift,
                          recv->args.comm.req,
                          req
                         );
            } else if (recv->type == DAG_COMPLETE) {
                do_step(recv);
            }
        }

        NBC_DEBUG(10, "all recv started\n");

        /* Append all entry points to a list polled during progress */
        OPAL_THREAD_LOCK(&req->active_list_lock);
        for (int i = req->n_ready_nodes; i < req->n_entry_nodes; ++i) {
            pcoll_dag_item * recv = req->entry_nodes[i];
            if ( DAG_COMPLETE != recv->type) {
                opal_list_append(&req->active_nodes, &recv->super.super);
            }
        }
        OPAL_THREAD_UNLOCK(&req->active_list_lock);

        /* Register the request for progress */
        OPAL_THREAD_LOCK(&pcoll_list_lock);
        opal_list_append(&pending_pcoll_requests, &req->req_ompi.super.super);
        OPAL_THREAD_UNLOCK(&pcoll_list_lock);

        NBC_DEBUG(10, "req %p pending for progress\n", req);
    }

    NBC_DEBUG(10,"%ld req started\n", count);

    return OMPI_SUCCESS;
}


/*
 * This routine set a partition as ready for send.
 * This means the data is ready and will not change until collective completion
 * Called on MPI_Pready or MPI_Pready_range.
 */
static int
ompi_mca_coll_bullnbc_pready_dag(unsigned long min, unsigned long max,
                                 ompi_request_t *request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request;
    NBC_DEBUG(10,"parts %ld -> %ld ready\n", min, max);
    if (max >= req->total_sparts) {
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Too high partition", false,
                       max, "palltoallr", req->total_sparts -1);
        max = req->total_sparts -1;
    }

    /* Take the lock in case a dependency needs to add polling tasks */
    OPAL_THREAD_LOCK(&req->active_list_lock);
    for (unsigned long part = min; part <= max; ++part){
        do_step(req->entry_nodes[part]);
    }
    OPAL_THREAD_UNLOCK(&req->active_list_lock);

    return OMPI_SUCCESS;
}

/*
 * This routine test if a range of partitioned has been received.
 * This is the partition equivalent of MPI_Testall
 * Called on MPI_Parrived,
 */
static int
ompi_mca_coll_bullnbc_parrived_dag(unsigned long min, unsigned long max,
                                   int*flag, ompi_request_t *request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request;
    int arrived = 1;

    if (0 == req->total_rparts) {
        fprintf(stderr, "No recv request expected on this collective\n"); fflush(stderr);
        goto failed;
    }
    if (max >= req->total_rparts) {
        opal_show_help("help-mpi-coll-bullnbc.txt",
                       "Too high partition", false,
                       max, "collective", req->total_rparts -1);
        max = req->total_sparts -1;
    }
    for (unsigned long part = min; part <= max; ++part){
        arrived = REQUEST_COMPLETE(req->user_part_reqs[req->total_sparts + part]);
        if (!arrived){
            opal_progress();
            break;
        }
    }

    *flag = arrived;
    NBC_DEBUG(10,"Arrived %d-%d = %d\n", min, max, arrived);

    return OMPI_SUCCESS;

failed:
    *flag = 0;
    return OMPI_ERR_BAD_PARAM;
}

/* Debug routine that runs recursively through all the DAG items to check that
 * nodes are in a valid state */
void
recursive_item_check(pcoll_dag_item* item)
{
    if(NULL == item) {
        abort();
    }
    if (item->backward_deps != item->current_unlock){
        NBC_DEBUG(1, "itm %p\nERROR still in use : %d/%dcompleted \n",
                  item, item->current_unlock, item->backward_deps);
        abort();
    }
    if (item->n_next_steps == 1) {
            recursive_item_check(item->next_step);
    } else if (item->n_next_steps > 1) {
        for ( int i = 0; i < item->n_next_steps; ++i) {
                recursive_item_check(item->next_steps[i]);
        }
    }
}


/* Routine to free a DAG itemand recurvely its dependencies
 * Called on all entry points during request free */
void recursive_item_free(pcoll_dag_item* item, const int expected_use)
{
    if(NULL == item) {
        NBC_DEBUG(5, "itm %p\nERROR cannot free \n", item);
        return;
    }
    if (item->type == DAG_CONVERT) {
        if (!ompi_datatype_is_predefined(item->args.convert.in_ddt)) {
            OBJ_RELEASE(item->args.convert.in_ddt);
        }
        if (!ompi_datatype_is_predefined(item->args.convert.out_ddt)) {
            OBJ_RELEASE(item->args.convert.out_ddt);
        }
    } else if (item->type == DAG_OP) {
        if (!ompi_datatype_is_predefined(item->args.op.ddt)) {
            OBJ_RELEASE(item->args.op.ddt);
        }
    }
    if( item->n_started != expected_use){
        NBC_DEBUG(1, "Item %p run %d time (expected %d)", item, item->n_started, expected_use);
        abort();
    }


   if (item->n_next_steps > 0) {
        for ( int i = 0; i < item->n_next_steps; ++i) {
            int32_t refcount = opal_atomic_add_fetch_32(&item->next_steps[i]->backward_deps, -1);
            if (0 == refcount) {
                NBC_DEBUG(10, "%p -> %d/%d free(%p)\n",
                          item, i,item->n_next_steps, item->next_steps[i]);
                recursive_item_free(item->next_steps[i], expected_use);
            }
        }
        free(item->next_steps);
    }

    free(item);
}

int dag_free_user_subreqs(ompi_coll_bullnbc_pcoll_request_t *req)
{

    int n_user_subreqs = req->total_sparts + req->total_rparts;
    for (int i = 0; i < n_user_subreqs; ++i) {
        OMPI_REQUEST_FINI(req->user_part_reqs[i]);
        opal_free_list_return(&part_req_free_list, &req->user_part_reqs[i]->super);
    }
    free(req->user_part_reqs);
    req->user_part_reqs = NULL;
}

/*
 * Free a request scheduled with a DAG
 */
int
ompi_mca_coll_bullnbc_free_dag (ompi_request_t** request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)*request;

        if (!REQUEST_COMPLETE(&req->req_ompi)){
            NBC_DEBUG(1, "Cannot free collnot completed\n");
        }
    //for(int i=0; i<req->n_entry_nodes;++i){
    //    recursive_item_check(req->entry_nodes[i]);
    //}

    for(int i=0; i<req->n_entry_nodes;++i){
        recursive_item_free(req->entry_nodes[i], req->n_started);
    }
    for(int i=0; i<req->n_created_ddt;++i){
        ompi_datatype_destroy(&req->created_ddt[i]);
    }
    free(req->created_ddt);

    free(req->entry_nodes);
    NBC_DEBUG(10, "Free tmpbuf %p of %p \n", req->tmpbuf, req);
    free(req->tmpbuf);

    free(req->internal_reqs);
    req->internal_reqs = NULL;

    dag_free_user_subreqs(req);

    OBJ_DESTRUCT(&req->active_nodes);
    OBJ_DESTRUCT(&req->active_list_lock);
    if (req->win) {
        NBC_DEBUG(5, "Win free %p\n", req->win);
        ompi_win_free(req->win);
        req->win= NULL;
    }

    NBC_DEBUG(5, "Free %p\n", request);
    opal_free_list_return(&mca_coll_bullnbc_pcoll_requests, &req->req_ompi.super);

    return OMPI_SUCCESS;
}

/*
 * Alloc an empty request in order to schedule a DAG
 */
ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_alloc_pcoll_request_dag (void)
{
    ompi_coll_bullnbc_pcoll_request_t* req;

    req = (ompi_coll_bullnbc_pcoll_request_t*) opal_free_list_wait(&mca_coll_bullnbc_pcoll_requests);

    OBJ_CONSTRUCT(&req->active_nodes, opal_list_t);

    OMPI_REQUEST_INIT(&req->req_ompi, true);
    req->req_ompi.req_start = ompi_mca_coll_bullnbc_dag_start_dag;
    req->req_ompi.req_ready = ompi_mca_coll_bullnbc_pready_dag;
    req->req_ompi.req_arrived = ompi_mca_coll_bullnbc_parrived_dag;
    req->req_ompi.req_free = ompi_mca_coll_bullnbc_free_dag;
    req->req_ompi.req_cancel = NULL;
    req->req_ompi.req_type = OMPI_REQUEST_PART;
    req->progress_cb = ompi_mca_coll_bullnbc_pcoll_update_dag;
    OBJ_CONSTRUCT(&req->active_list_lock, opal_mutex_t);

    req->n_started = 0;
    req->n_complete = 0;
    req->n_created_ddt = 0;
    req->created_ddt = NULL;
    req->tmpbuf = NULL;

    /* Requests for internal comms */
    req->n_internal_reqs = 0;
    req->n_entry_sched = 0;
    req->n_internal_reqs_sched = 0;
    req->internal_reqs = NULL;

    req->win = NULL;

    return req;
}

/*
 * Alloc user part requests and expose them to user */
int
dag_alloc_user_subreqs(ompi_coll_bullnbc_pcoll_request_t* req,
                       size_t total_sparts,
                       ompi_request_t** user_sreqs,
                       size_t total_rparts,
                       ompi_request_t** user_rreqs)
{
    int n_user_subreqs = total_sparts + total_rparts;

    req->total_sparts = total_sparts;
    req->user_sreqs = user_sreqs;
    req->total_rparts = total_rparts;
    req->user_rreqs = user_rreqs;
    req->user_part_reqs = malloc(n_user_subreqs * sizeof (ompi_request_t*));

    NBC_DEBUG(10, "user_part_reqs %p init with %d (%d+%d) entries \n",
              req->user_part_reqs, n_user_subreqs, req->total_sparts,req->total_rparts);

    for (int i = 0; i < n_user_subreqs ; ++i) {
        req->user_part_reqs[i] = (ompi_request_t*)opal_free_list_wait(&part_req_free_list);
        /* Other fields were initialized by the constructor for ompi_request_t */
        OMPI_REQUEST_INIT(req->user_part_reqs[i], true);
        req->user_part_reqs[i]->req_type = OMPI_REQUEST_SUBPART;
    }

    if (MPIX_NO_REQUESTS != req->user_sreqs) {
        memcpy(req->user_sreqs,
               req->user_part_reqs,
               req->total_sparts * sizeof(ompi_request_t*));
    }
    if (MPIX_NO_REQUESTS != req->user_rreqs) {
        memcpy(req->user_rreqs,
               req->user_part_reqs + req->total_sparts,
               req->total_rparts * sizeof(ompi_request_t*));
    }

    return OMPI_SUCCESS;
}

/*
 * Alloc and initialize a DAG request */
ompi_coll_bullnbc_pcoll_request_t*
coll_bullnbc_pcoll_init_dag_ntag_req(size_t total_sparts,
                                     ompi_request_t** user_sreqs,
                                     size_t total_rparts,
                                     ompi_request_t** user_rreqs,
                                     unsigned int n_internal_reqs,
                                     unsigned int n_recv,
                                     struct ompi_communicator_t *comm,
                                     int ntags,
                                     ompi_request_t ** user_main_req)
{
    ompi_coll_bullnbc_pcoll_request_t * req;
    req = ompi_mca_coll_bullnbc_alloc_pcoll_request_dag();
    req->tag = ompi_coll_base_nbc_reserve_tags(comm, ntags);
    *user_main_req = &req->req_ompi;

    req->n_internal_reqs = n_internal_reqs;

    dag_alloc_user_subreqs(req, total_sparts, user_sreqs, total_rparts, user_rreqs);

    req->n_ready_nodes = total_sparts;
    req->n_entry_nodes = total_sparts + n_recv;
    req->entry_nodes = malloc(req->n_entry_nodes * sizeof(pcoll_dag_item*));
    req->internal_reqs = malloc(req->n_internal_reqs * sizeof(ompi_request_t *));
    if (NULL == req->internal_reqs) {
        NBC_Error("Failed to malloc (%s,%d)\n",__func__,__LINE__);
    }

    return req;
}
/* Alloc and initialize a DAG request */
ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_pcoll_init_dag_req(size_t total_sparts,
                                         ompi_request_t** user_sreqs,
                                         size_t total_rparts,
                                         ompi_request_t** user_rreqs,
                                         unsigned int n_internal_reqs,
                                         unsigned int n_recv,
                                         struct ompi_communicator_t *comm,
                                         ompi_request_t ** user_main_req)
{
    return coll_bullnbc_pcoll_init_dag_ntag_req (total_sparts, user_sreqs,
                                                 total_rparts, user_rreqs,
                                                 n_internal_reqs, n_recv,
                                                 comm, 1, user_main_req);
}

/* Decrement a lock on a DAG item and execute it if the lock reach 0 */
static void
unlock(pcoll_dag_item* item) {
    int32_t lock = opal_atomic_add_fetch_32(&item->current_unlock, -1);
    if (lock){
        NBC_DEBUG(5, " %p locked %d/%d\n", item, lock, item->backward_deps);
        return;
    }
    /* Reset deps */
    item->current_unlock = item->backward_deps;
    do_step(item);
}

/* Execute a DAG item and unlock dependencies of the execution suceeds */
static inline int
do_step(pcoll_dag_item* item) {
#if PROFILE
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    switch (item->type){
        case DAG_CONVERT:
            NBC_DEBUG(5, "Move %p -> %p\n",
                      item->args.convert.inbuf,
                      item->args.convert.outbuf);
            ompi_datatype_sndrcv(item->args.convert.inbuf,
                                 item->args.convert.in_count,
                                 item->args.convert.in_ddt,
                                 item->args.convert.outbuf,
                                 item->args.convert.out_count,
                                 item->args.convert.out_ddt);
            break;
        case DAG_COPY:
            NBC_DEBUG(5, "COPY %p\n", item);
            memcpy(item->args.copy.dst,
                   item->args.copy.src,
                   item->args.copy.size);
            break;
        case DAG_OP:
            NBC_DEBUG(5, "OP %p (%d ddt)\n", item, item->args.op.size);
            if(item->args.op.size) {
                NBC_DEBUG(10, "OP (%p) %d +> %d (%p)\n",
                          item->args.op.src,
                          ((char*)item->args.op.src)[0],
                          ((char*)item->args.op.dst)[0],
                          item->args.op.dst);
                ompi_op_reduce(item->args.op.op,
                               (void*) item->args.op.src,
                               item->args.op.dst,
                               item->args.op.size,
                               item->args.op.ddt);
                NBC_DEBUG(10, "OP = %d\n", item->args.op.dst[0]);
            }
            break;
        case DAG_SEND:
            ;
            ompi_coll_bullnbc_pcoll_request_t * req;
            req = item->args.comm.collreq;
            if (! item->args.comm.started){
                int tag = item->args.comm.collreq->tag;
                tag += item->args.comm.tagshift;
                MCA_PML_CALL(isend(item->args.comm.buf,
                                   item->args.comm.count,
                                   item->args.comm.datatype,
                                   item->args.comm.peer,
                                   tag,
                                   MCA_PML_BASE_SEND_STANDARD,
                                   item->args.comm.comm,
                                   item->args.comm.req));
                req->n_internal_reqs_started++;
                NBC_DEBUG(5, "SEND start %p (dst %d, tag %d, %d ddt %p "
                          "pml_req %p) from collreq %p\n",
                          item, item->args.comm.peer,
                          tag, item->args.comm.count,
                          item->args.comm.datatype,
                          item->args.comm.req,
                          item, req);
                if (!REQUEST_COMPLETE(*item->args.comm.req)){
                    item->args.comm.started = 1;
                    opal_list_append(&req->active_nodes, &item->super.super);
                    return OMPI_ERR_RESOURCE_BUSY;
                }
            } else if (!REQUEST_COMPLETE(*item->args.comm.req)){
                NBC_DEBUG(20, "SEND %p pending...",item);
                return OMPI_ERR_RESOURCE_BUSY;
            }

            item->args.comm.started = 0;
            if (mca_coll_bullnbc_component.debug_read_user_data && item->args.comm.count) {
                NBC_DEBUG(5, "SEND %p completed @%p (%dobj: %d ..) to %d\n",
                          item, item->args.comm.buf,
                          item->args.comm.count,
                          ((int*)item->args.comm.buf)[0],
                          item->args.comm.peer);
            } else {
                NBC_DEBUG(5, "SEND %p completed \n", item);
            }
            break;
        case DAG_RECV:
            if (!REQUEST_COMPLETE(*item->args.comm.req)){
                return OMPI_ERR_RESOURCE_BUSY;
            }

            if (mca_coll_bullnbc_component.debug_read_user_data && item->args.comm.count) {
                NBC_DEBUG(5, "RECV %p (internal %p) completed addr %p (%dobj: %d ..) from %d\n",
                          item, item->args.comm.req,
                          item->args.comm.buf,
                          item->args.comm.count,
                          ((int*)item->args.comm.buf)[0],
                          item->args.comm.peer);
            } else {
                NBC_DEBUG(5, "RECV %p (internal %p) completed\n",
                          item, item->args.comm.req);
            }
            break;
#if OMPI_MPI_NOTIFICATIONS
        case DAG_PUT:
            NBC_DEBUG(5, "PUT %p\n", item);
            NBC_DEBUG(5, "Exec put %d of %d elems at offset +%d on win %p\n",
        item->args.put.notif_id, item->args.put.rcount,item->args.put.disp,item->args.put.win);
            ompi_put_notify(item->args.put.buf,
                            item->args.put.scount,
                            item->args.put.sddt,
                            item->args.put.dst,
                            //0,
                            item->args.put.disp,
                            item->args.put.rcount,
                            item->args.put.rddt,
                            item->args.put.win,
                            item->args.put.notif_id);
            break;
        case DAG_NOTIF:
            ; int flag = 0;
            static uint32_t reentrance = 0;
            /* Win_test call progress if not found, the same notif wait;
             * may lead to stack overflow calling itself in a loop */
            if (opal_atomic_swap_32(&reentrance,1)) {
                return OMPI_ERR_RESOURCE_BUSY;
            }
            NBC_DEBUG(5, "Wait notif %d on window %p\n", item->args.notif.id, item->args.notif.win);
            if (OMPI_SUCCESS !=ompi_win_test_notify(item->args.notif.win,
                                                    item->args.notif.id,
                                                    &flag)) {
                NBC_Error("Fail to test notif\n");abort();
            }
            reentrance = 0;
            if (!flag) {
                return OMPI_ERR_RESOURCE_BUSY;
            }
            /* Just execute deps */
            NBC_DEBUG(5, "NOTIF DONE %p\n", item);
            break;
#endif
        case DAG_READY:
            NBC_DEBUG(5, "READY DONE %p\n", item);
            /* Just execute deps */
            break;
        case DAG_COMPLETE:
            ;ompi_request_t * subreq = item->args.complete.req;
            NBC_DEBUG(5, "COMPLETE %p -> %p\n", item, subreq);
            subreq->req_status.MPI_ERROR = OMPI_SUCCESS;
            ompi_request_complete(subreq, true);
            ompi_coll_bullnbc_pcoll_request_t * collreq = item->args.complete.collreq;
            opal_atomic_add_fetch_32(&collreq->n_completed_subreqs, 1);
            break;
        default:
            NBC_Error("Unexpected type %d for dag item %p \n", item->type, item);
            return OMPI_ERROR;
    }

#if PROFILE
    clock_gettime(CLOCK_MONOTONIC, &end);
    profiling[item->type] += (end.tv_sec-start.tv_sec)*1e9 + (end.tv_nsec- start.tv_nsec);
    n_profiled[item->type] ++;
#endif

    item->n_started++;
    /* This item is complete, dependancies can be started */
    if (item->n_next_steps) {
        for (int i = 0; i < item->n_next_steps; ++i){
            NBC_DEBUG(5, "%s %p ==%d/%d==> %s %p\n",
                      dag_item_type_to_str(item->type), item,
                      i, item->n_next_steps,
                      dag_item_type_to_str(item->next_steps[i]->type), item->next_steps[i]);
            unlock(item->next_steps[i]);
        }
    } else {
        NBC_DEBUG(5, "%p has NO DEPS\n", item);
    }

    return OMPI_SUCCESS;
}


/*
 * Progress routine for a DAG request.
 * Call progress on entry points and try to complete the collective request.
 * Called under pcoll_list lock */
int
ompi_mca_coll_bullnbc_pcoll_update_dag (ompi_coll_bullnbc_pcoll_request_t*req)
{

    int count = 0;
    pcoll_dag_item* entry_point, *next;
    OPAL_THREAD_LOCK(&req->active_list_lock);
    OPAL_LIST_FOREACH_SAFE (entry_point, next, &req->active_nodes, pcoll_dag_item) {
        int ret = do_step(entry_point);
        next = (pcoll_dag_item *) entry_point->super.super.opal_list_next ;
        if (OMPI_SUCCESS == ret) {
            count ++;
            opal_list_remove_item(&req->active_nodes,
                                  &entry_point->super.super);
        }
    }

    OPAL_THREAD_UNLOCK(&req->active_list_lock);
    if ( count && !opal_list_get_size(&req->active_nodes)) {
        NBC_DEBUG(5, "All entry point done\n");
    }

    int to_complete = req->total_sparts + req->total_rparts;
    if (req->n_completed_subreqs != to_complete){
        NBC_DEBUG(20,"Coll partitioned %p to complete %d/%d\n",
                  req, req->n_completed_subreqs, to_complete);
        return count;
    }

    if (req->n_internal_reqs_started < req->n_internal_reqs) {
        /* Some requests are not initialized */
        NBC_DEBUG(10, "Only %d/%d internal comm started,"
                  " no way the request %p is complete ...\n",
                  req->n_internal_reqs_started, req->n_internal_reqs, req);
        return count;
    }

    int internal_complete = 0;
    for (int i = 0; i < req->n_internal_reqs; ++i) {
        if (REQUEST_COMPLETE(req->internal_reqs[i])){
            internal_complete ++;
            NBC_DEBUG(50,"internal partitioned %d %p completed\n",
                      i, req->internal_reqs[i]);
        } else {
            NBC_DEBUG(50,"internal partitioned %d %p pending\n",
                      i, req->internal_reqs[i]);
        }
    }
    if (internal_complete < req->n_internal_reqs){
        NBC_DEBUG(20,"Coll partitioned %p pt2pt not all completed %d/%d\n",
                  req, internal_complete, req->n_internal_reqs);

        return count;
    }

    NBC_DEBUG(10,"Coll partitioned %p completed %d/%d\n",
              req, count, to_complete);

    req->n_complete++;
    if (req->n_complete != req->n_started){
        NBC_DEBUG(1, "Req %p started %d time/ ended %d time\n",
                  req,req->n_started, req->n_complete);
    }

    opal_list_remove_item(&pending_pcoll_requests,
                          &req->req_ompi.super.super);
    req->req_ompi.req_status.MPI_ERROR = OMPI_SUCCESS;
    ompi_request_complete(&(req->req_ompi), true );

    return count;
}
