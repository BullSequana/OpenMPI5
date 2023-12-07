/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/**
 * @file
 *
 * This file provides an implementation for noop partitioned collective requests
 */

#include "ompi/request/request.h"
#include "ompi/datatype/ompi_datatype.h"
#include "coll_bullnbc_partitioned.h"

static int ompi_mca_coll_bullnbc_noop_start(unsigned long count, ompi_request_t **requests)
{
    ompi_coll_bullnbc_pcoll_request_t *req;
    for (int req_idx = 0; req_idx < count; ++req_idx) {
        ompi_request_t *request = requests[req_idx];
        req = (ompi_coll_bullnbc_pcoll_request_t *) request;
        int n_user_subreqs = req->total_sparts + req->total_rparts;

        for (int subreq_idx = 0; subreq_idx < n_user_subreqs; ++subreq_idx) {
            ompi_request_t *subreq = req->user_part_reqs[subreq_idx];
            subreq->req_state = OMPI_REQUEST_ACTIVE;
            subreq->req_status.MPI_ERROR = OMPI_SUCCESS;
            subreq->req_complete = REQUEST_COMPLETED;
        }

        request->req_status.MPI_ERROR = OMPI_SUCCESS;
        request->req_complete = REQUEST_COMPLETED;
    }
    return OMPI_SUCCESS;
}
static int
ompi_mca_coll_bullnbc_noop_pready(unsigned long min, unsigned long max,
                                 ompi_request_t *request) {
    return OMPI_SUCCESS;
}
static int
ompi_mca_coll_bullnbc_parrived_noop(unsigned long min, unsigned long max,
                                   int*flag, ompi_request_t *request) {
    *flag = 1;
    return OMPI_SUCCESS;
}
static int
ompi_mca_coll_bullnbc_free_noop(ompi_request_t** request)
{
    ompi_coll_bullnbc_pcoll_request_t* req = (ompi_coll_bullnbc_pcoll_request_t*)*request;
    dag_free_user_subreqs(req);
    opal_free_list_return(&mca_coll_bullnbc_pcoll_requests, &req->req_ompi.super);
    return OMPI_SUCCESS;
}

ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_pcoll_init_empty_req(size_t total_sparts,
                                               ompi_request_t** user_sreqs,
                                               size_t total_rparts,
                                               ompi_request_t** user_rreqs,
                                               struct ompi_communicator_t *comm,
                                               ompi_request_t ** user_main_req)
{
    ompi_coll_bullnbc_pcoll_request_t* req;

    req = (ompi_coll_bullnbc_pcoll_request_t*) opal_free_list_wait(&mca_coll_bullnbc_pcoll_requests);
    OMPI_REQUEST_INIT(&req->req_ompi, true);
    req->req_ompi.req_start = ompi_mca_coll_bullnbc_noop_start;
    req->req_ompi.req_ready = ompi_mca_coll_bullnbc_noop_pready;
    req->req_ompi.req_arrived = ompi_mca_coll_bullnbc_parrived_noop;
    req->req_ompi.req_free = ompi_mca_coll_bullnbc_free_noop;
    req->req_ompi.req_cancel = NULL;
    req->req_ompi.req_type = OMPI_REQUEST_PART;
    req->progress_cb = NULL;

    req->total_sparts = total_sparts;
    req->user_sreqs = user_sreqs;
    req->total_rparts = total_rparts;
    req->user_rreqs = user_rreqs;
    dag_alloc_user_subreqs(req, total_sparts, user_sreqs, total_rparts, user_rreqs);

    *user_main_req = &req->req_ompi;

    return req;
}

