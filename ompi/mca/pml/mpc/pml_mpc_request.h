/*
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef PML_MPC_REQUEST_H
#define PML_MPC_REQUEST_H

#include "ompi/request/request.h"
#include "opal/class/opal_free_list.h"

#include "mpc_lowcomm.h"
#include "pml_mpc.h"

typedef enum {
    PML_MPC_NO_ARGS,
    PML_MPC_ISEND_ARGS,
    PML_MPC_IRECV_ARGS
} pml_mpc_msg_args_t;

typedef struct pml_mpc_isend_args_t {
    const void* buf;
    size_t count;
    ompi_datatype_t* datatype;
    int dst;
    int tag;
    mca_pml_base_send_mode_t mode;
    ompi_communicator_t* comm;
} pml_mpc_isend_args_t;

typedef struct pml_mpc_irecv_args_t {
    void* buf;
    size_t count;
    ompi_datatype_t* datatype;
    int src;
    int tag;
    struct ompi_communicator_t* comm;
} pml_mpc_irecv_args_t;

typedef struct pml_mpc_request_t pml_mpc_request_t;
OMPI_DECLSPEC OBJ_CLASS_DECLARATION(pml_mpc_request_t);
struct pml_mpc_request_t {
    ompi_request_t ompi_req;
    opal_list_item_t pending_list_item;
    mpc_lowcomm_request_t mpc_req;
    pml_mpc_msg_args_t has_args;
    union {
        pml_mpc_isend_args_t isend;
        pml_mpc_irecv_args_t irecv;
    } args;
    void* tmp_buf;           // for bufferization
    bool irecv_needs_unpack; // after irecv non contiguous dtt
    bool buffered;           // Use to free attached buffer allocation in Bsend
};

// PML requests callback
int
pml_mpc_request_free_fn(struct ompi_request_t** rptr);
int
pml_mpc_request_complete_fn(struct ompi_request_t* request);
int
pml_mpc_request_cancel_fn(struct ompi_request_t* request, int flag);

// TODO: use once it can worl in MPC
/* // MPC callback, called when MPC request completes */
/* int */
/* pml_mpc_request_completion_callback(mpc_lowcomm_request_t* mpc_reqp); */

// initialize
static inline pml_mpc_request_t*
pml_mpc_request_alloc(void)
{
    pml_mpc_request_t* req = (pml_mpc_request_t*)opal_free_list_get(
                &mca_pml_mpc_component.requests_free_list);
    pml_mpc_verbose(30, "request_alloc: %p", (void*)req);
    return req;
}
static inline void
pml_mpc_request_free(pml_mpc_request_t* request)
{
    pml_mpc_verbose(31, "request_free: %p", (void*)request);
    opal_free_list_return(&mca_pml_mpc_component.requests_free_list,
                          &request->ompi_req.super);
}

static inline void
pml_mpc_request_init(pml_mpc_request_t* req, bool persistant)
{
    pml_mpc_verbose(31, "request_init: %p", (void*)req);
    OBJ_CONSTRUCT(&req->ompi_req, ompi_request_t); //  are those two necessary
    OMPI_REQUEST_INIT(&req->ompi_req, persistant);
    req->ompi_req.req_type = OMPI_REQUEST_PML;
    req->ompi_req.req_state = OMPI_REQUEST_ACTIVE; // this is what makes waiting
    req->ompi_req.req_status = ompi_status_empty;
    req->ompi_req.req_complete = REQUEST_PENDING;
    req->ompi_req.req_free = pml_mpc_request_free_fn;
    req->ompi_req.req_complete_cb = pml_mpc_request_complete_fn;
    req->ompi_req.req_cancel = pml_mpc_request_cancel_fn;
    req->has_args = PML_MPC_NO_ARGS;
    req->tmp_buf = NULL; // not buffered
    req->irecv_needs_unpack = false;
    req->buffered = false;
}

static inline void
pml_mpc_request_fini(pml_mpc_request_t* req)
{
    pml_mpc_verbose(31, "request_fini: %p", (void*)req);
    req->irecv_needs_unpack = false;
    if (NULL != req->tmp_buf) {
        if(req->buffered) {
            mca_pml_base_bsend_request_free((void *) req->tmp_buf);
        } else {
            free(req->tmp_buf);
        }
    }
    OMPI_REQUEST_FINI(&req->ompi_req);
    OBJ_DESTRUCT(&req->ompi_req);
}

// helpers for containers of request fields pointers, container_of equivalents
static inline pml_mpc_request_t*
pml_mpc_request_of_mpc_req(mpc_lowcomm_request_t* mpc_reqp)
{
    return (pml_mpc_request_t*)((char*)mpc_reqp
                                - offsetof(pml_mpc_request_t, mpc_req));
}
static inline pml_mpc_request_t*
pml_mpc_request_of_pending_list_item(opal_list_item_t* item)
{
    return (pml_mpc_request_t*)((char*)item
                                - offsetof(pml_mpc_request_t,
                                            pending_list_item));
}
static inline pml_mpc_request_t*
pml_mpc_request_of_ompi_req(ompi_request_t* ompi_req)
{
    return (pml_mpc_request_t*)((char*)ompi_req
                                - offsetof(pml_mpc_request_t, ompi_req));
}

// Pending list

void
pml_mpc_pending_list_init(void);
void
pml_mpc_pending_list_fini(void);
void
pml_mpc_pending_list_append(pml_mpc_request_t* req);
int
pml_mpc_pending_list_size(void);
pml_mpc_request_t*
pml_mpc_pending_list_remove_first(void);

// free-list
void
pml_mpc_requests_free_list_init(void);
void
pml_mpc_requests_free_list_fini(void);

// start persistant requests
int
mca_pml_mpc_isend_start(pml_mpc_request_t* req);
int
mca_pml_mpc_irecv_start(pml_mpc_request_t* req);

// adding status here
void
pml_mpc_status_convert(mpc_lowcomm_status_t* mpc_status,
                       ompi_status_public_t* status);
#endif
