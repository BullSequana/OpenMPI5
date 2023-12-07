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
 * based on partitioned point-to-point comunications.
 * This file provides implementation for request creation and free,
 * and for start, pready, parrived and progress functions.
 *
 */

#include "coll_bullnbc.h"
#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_internal.h"

opal_free_list_t mca_coll_bullnbc_pcoll_requests = {{{0}}};
opal_list_t pending_pcoll_requests = {0};
opal_mutex_t pcoll_list_lock = {0};

opal_free_list_t subpart_free_list = {0};
opal_free_list_t part_req_free_list = {0};

int ompi_mca_coll_bullnbc_pcoll_update (ompi_coll_bullnbc_pcoll_request_t*req);
static void subpart_requests_free(int count, ompi_request_t ** reqs);

static int
ompi_mca_coll_bullnbc_start(unsigned long count, ompi_request_t **request)
{
    for (unsigned long ireq = 0; ireq < count; ++ireq){
        ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request[ireq];

        if (MPIX_NO_REQUESTS != req->user_sreqs) {
            /* Try to complete send sub requests */
            for (size_t part=0; part < req->total_sparts; ++part){
                ompi_request_t*user_part_req = req->user_sreqs[part];
                user_part_req->req_state = OMPI_REQUEST_ACTIVE;
                user_part_req->req_complete = REQUEST_PENDING;
            }
        }
        if (MPIX_NO_REQUESTS != req->user_rreqs) {
            /* Try to complete recv sub requests */
            for (size_t part=0; part < req->total_rparts; ++part){
                ompi_request_t*user_part_req = req->user_rreqs[part];
                user_part_req->req_state = OMPI_REQUEST_ACTIVE;
                user_part_req->req_complete = REQUEST_PENDING;
            }
        }


        for (int i = 0; i <  req->n_internal_reqs; ++i) {
            req->internal_reqs[i]->req_start(1, &req->internal_reqs[i]);
            NBC_DEBUG(10,"req%ld-%d started\n", ireq,i);
        }

        req->req_ompi.req_state = OMPI_REQUEST_ACTIVE;
        req->req_ompi.req_status.MPI_TAG = MPI_ANY_TAG;
        req->req_ompi.req_status.MPI_ERROR = OMPI_SUCCESS;
        req->req_ompi.req_status._cancelled = 0;
        OPAL_ATOMIC_SWAP_PTR(&req->req_ompi.req_complete, REQUEST_PENDING);

        OPAL_THREAD_LOCK(&pcoll_list_lock);
        opal_list_append(&pending_pcoll_requests, &req->req_ompi.super.super);
        OPAL_THREAD_UNLOCK(&pcoll_list_lock);
    }
    NBC_DEBUG(10,"%ld req started\n", count);

    return OMPI_SUCCESS;
}

static int
ompi_mca_coll_bullnbc_pready(unsigned long min, unsigned long max, ompi_request_t *request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request;

    for (unsigned long part = min; part <= max; ++part){
        ompi_coll_bullnbc_subpart *subpart;
        OPAL_LIST_FOREACH(subpart, &req->send_subparts[part], ompi_coll_bullnbc_subpart){
            subpart->sub_req->req_ready(subpart->part_idx,
                                        subpart->part_idx,
                                        subpart->sub_req);
            NBC_DEBUG(10, "ready %ld : forward to subpart %p - %ld\n",
                      part, subpart->sub_req, subpart->part_idx);
        }
    }
    return OMPI_SUCCESS;
}

static int
ompi_mca_coll_bullnbc_parrived(unsigned long min, unsigned long max, int*flag, ompi_request_t *request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)request;
    int arrived = 1;

    for (unsigned long part = min; part <= max; ++part){
        ompi_coll_bullnbc_subpart *subpart;
        OPAL_LIST_FOREACH(subpart, &req->recv_subparts[part], ompi_coll_bullnbc_subpart){
            subpart->sub_req->req_arrived(subpart->part_idx,
                                          subpart->part_idx,
                                          &arrived, subpart->sub_req);
            if (!arrived){
                goto for_exit;
            }
        }
    }

for_exit:
    *flag = arrived;
    NBC_DEBUG(10,"Arrived %d-%d = %d\n", min, max, arrived);

    return OMPI_SUCCESS;
}

int
ompi_mca_coll_bullnbc_free (ompi_request_t** request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)*request;

    subpart_requests_free(req->total_sparts, req->user_sreqs);
    subpart_requests_free(req->total_rparts, req->user_rreqs);


    /* Free all descriptions of internal send partitions */
    if (NULL != req->send_subparts){
        for (unsigned long part = 0; part < req->total_sparts; ++part){
            ompi_coll_bullnbc_subpart *subpart, *next;
            OPAL_LIST_FOREACH_SAFE(subpart, next, &req->send_subparts[part], ompi_coll_bullnbc_subpart){
                opal_list_remove_item(&req->send_subparts[part], &subpart->super.super);
                opal_free_list_return(&subpart_free_list, &subpart->super);
            }
            OBJ_DESTRUCT(&req->send_subparts[part]);
        }
    }

    /* Free all descriptions of internal recv partitions */
    if (NULL != req->recv_subparts){
        for (unsigned long part = 0; part < req->total_rparts; ++part){
            ompi_coll_bullnbc_subpart *subpart, *next;
            OPAL_LIST_FOREACH_SAFE(subpart, next, &req->recv_subparts[part], ompi_coll_bullnbc_subpart){
                opal_list_remove_item(&req->recv_subparts[part], &subpart->super.super);
                opal_free_list_return(&subpart_free_list, &subpart->super);
            }
            OBJ_DESTRUCT(&req->recv_subparts[part]);
        }
    }

    for(int i=0; i<req->n_internal_reqs;++i){
        req->internal_reqs[i]->req_free(&req->internal_reqs[i]);
    }

    free(req->send_subparts);
    free(req->recv_subparts);
    free(req->internal_reqs);
    req->internal_reqs = NULL;
    opal_free_list_return(&mca_coll_bullnbc_pcoll_requests, &req->req_ompi.super);
    return OMPI_SUCCESS;
}

ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_alloc_pcoll_request (void)
{
    ompi_coll_bullnbc_pcoll_request_t* req;

    req = (ompi_coll_bullnbc_pcoll_request_t*) opal_free_list_wait(&mca_coll_bullnbc_pcoll_requests);

    OMPI_REQUEST_INIT(&req->req_ompi, true);
    req->req_ompi.req_start = ompi_mca_coll_bullnbc_start;
    req->req_ompi.req_ready = ompi_mca_coll_bullnbc_pready;
    req->req_ompi.req_arrived = ompi_mca_coll_bullnbc_parrived;
    req->req_ompi.req_free = ompi_mca_coll_bullnbc_free;
    req->req_ompi.req_cancel = NULL;
    req->req_ompi.req_type = OMPI_REQUEST_PART;

    req->progress_cb = ompi_mca_coll_bullnbc_pcoll_update;

    return req;
}

int ompi_mca_coll_bullnbc_pcoll_update (ompi_coll_bullnbc_pcoll_request_t*req)
{
    if (MPIX_NO_REQUESTS != req->user_sreqs) {
        /* Try to complet send sub requests */
        for (unsigned long part=0; part < req->total_sparts; ++part){
            ompi_request_t*user_part_req = req->user_sreqs[part];
    /* DAG nodes */
            if (REQUEST_COMPLETE(user_part_req)){
                continue;
            }
            ompi_coll_bullnbc_subpart* subpart;
            int complete = 1;
            OPAL_LIST_FOREACH (subpart, &req->send_subparts[part], ompi_coll_bullnbc_subpart){
                complete =  REQUEST_COMPLETE(subpart->sub_req);
                if(! complete) {
                    /* Found a subpart that prevent completion */
                    break;
                }
            }
            if (complete) {
                // No subpart pending
                user_part_req->req_status.MPI_ERROR = OMPI_SUCCESS;
                ompi_request_complete(user_part_req, true);
            }
        }
    }

    if (MPIX_NO_REQUESTS != req->user_rreqs) {
        /* Try to complet recv sub requests */
        for (unsigned long part=0; part < req->total_rparts; ++part){
            ompi_request_t*user_part_req = req->user_rreqs[part];
            if (REQUEST_COMPLETE(user_part_req)){
                continue;
            }

            // search any non completed frag
            ompi_coll_bullnbc_subpart* subpart;
            int complete = 1;
            OPAL_LIST_FOREACH (subpart, &req->recv_subparts[part], ompi_coll_bullnbc_subpart){
                if (!REQUEST_COMPLETE(subpart->sub_req)){
                    subpart->sub_req->req_arrived(subpart->part_idx,
                                               subpart->part_idx,
                                               &complete, subpart->sub_req);
                    if (!complete){
                        // Found a subpart that prevent to complete
                        break;
                    }
                }
            }
            if (complete) {
                // No subpart pending
                user_part_req->req_status.MPI_ERROR = OMPI_SUCCESS;
                ompi_request_complete(user_part_req, true);
            }
        }
    }

    int count = 0;
    for (int i = 0; i < req->n_internal_reqs; ++i) {
        if (REQUEST_COMPLETE(req->internal_reqs[i])){
            count ++;
        }
    }

    if (count == req->n_internal_reqs){
        NBC_DEBUG(1,"Coll partitioned %p completed\n", req);
        opal_list_remove_item(&pending_pcoll_requests, &req->req_ompi.super.super);
        req->req_ompi.req_status.MPI_ERROR = OMPI_SUCCESS;
        NBC_DEBUG(1, "Freeing main req %p\n", req);
        ompi_request_complete(&(req->req_ompi), true );
    }

    return count;
}


int
subpart_requests_create(ompi_coll_bullnbc_pcoll_request_t* req)
{
    if (MPIX_NO_REQUESTS != req->user_sreqs) {
        for (size_t part = 0; part < req->total_sparts; ++part) {
            ompi_request_t * user_part_req = (ompi_request_t*)opal_free_list_wait(&part_req_free_list);
            OMPI_REQUEST_INIT(user_part_req, true);
            /* Other fields were initialized by the constructor for ompi_request_t */
            user_part_req->req_type = OMPI_REQUEST_SUBPART;
            /* TODO  As these requests should be equivalent for user to Isend
             * requests, user may want to get MPI_Status or to call erroneously
             * MPI_Pready, MPI_Free.
             * These data and function pointer are currently missing. */

            req->user_sreqs[part] = user_part_req;

        }
    }
    if (MPIX_NO_REQUESTS != req->user_rreqs) {
        for (size_t part = 0; part < req->total_rparts; ++part) {
            ompi_request_t * user_part_req = (ompi_request_t*)opal_free_list_wait(&part_req_free_list);
            OMPI_REQUEST_INIT(user_part_req, true);
            /* Other fields were initialized by the constructor for ompi_request_t */
            user_part_req->req_type = OMPI_REQUEST_SUBPART;
            /* TODO  As these requests should be equivalent for user to Irecv
             * requests, user may want to get MPI_Status or to call erroneously
             * MPI_Parrived, MPI_Free.
             * These data and function pointer are currently missing. */

            req->user_rreqs[part] = user_part_req;

        }
    }
    return OMPI_SUCCESS;
}

static void
subpart_requests_free(int count, ompi_request_t ** reqs)
{
    if (MPIX_NO_REQUESTS == reqs){
        return;
    }
    for (int part =0; part < count; ++part){
        ompi_request_t* req = reqs[part];
        OMPI_REQUEST_FINI(req);
        opal_free_list_return(&part_req_free_list, &req->super);
        reqs[part]=NULL;
    }
}
