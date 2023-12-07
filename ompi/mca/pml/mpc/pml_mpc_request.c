/*
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "pml_mpc_request.h"
#include "assert.h"
#include "mpc_lowcomm.h"
#include "ompi/request/request.h"
#include "pml_mpc.h"
#include "pml_mpc_pack.h"

OBJ_CLASS_INSTANCE(pml_mpc_request_t,
                   ompi_request_t,
                   pml_mpc_request_init,
                   pml_mpc_request_fini);

// request lifetime

int
pml_mpc_request_free_fn(struct ompi_request_t** rptr)
{
    ompi_request_t* ompi_req = *rptr;
    pml_mpc_request_t* req = pml_mpc_request_of_ompi_req(ompi_req);
    pml_mpc_verbose(
                30, "free req: %p, ompi_req: %p", (void*)req, (void*)ompi_req);
    pml_mpc_request_fini(req);
    pml_mpc_request_free(req);
    *rptr = MPI_REQUEST_NULL; // avoids double free for some reason.
    return 0;
}

int
pml_mpc_request_complete_fn(struct ompi_request_t* request)
{
    pml_mpc_request_t* req = pml_mpc_request_of_ompi_req(request);

    // unpack arrived irecv
    if (req->irecv_needs_unpack) {
        if (NULL == req->tmp_buf || PML_MPC_IRECV_ARGS != req->has_args) {
            pml_mpc_error("req=%p completion: cannot unpack", (void*)request);
            return OMPI_ERROR;
        }
        void* usr_buf = req->args.irecv.buf;
        void* tmp_buf = req->tmp_buf;
        int count = req->args.irecv.count;
        ompi_datatype_t* datatype = req->args.irecv.datatype;
        int ret = mca_pml_mpc_unpack(usr_buf, tmp_buf, count, datatype);
        if (OMPI_SUCCESS != ret) {
            pml_mpc_error("complete_fn ompi_req %p failed unpack",
                          (void*)request);
        }
    }
    if (req->buffered) {
        mca_pml_base_bsend_request_free((void *) req->tmp_buf);
        req->tmp_buf = NULL;
        req->buffered=false;
    }
    // fill request status
    mpc_lowcomm_status_t mpc_status;
    mpc_lowcomm_commit_status_from_request(&req->mpc_req, &mpc_status);
    pml_mpc_status_convert(&mpc_status, &req->ompi_req.req_status);

    pml_mpc_verbose(30, "complete_fn ompi_req %p", (void*)request);
    return OMPI_SUCCESS;
}
int
pml_mpc_request_cancel_fn(struct ompi_request_t* request, int flag)
{
    (void)flag; // TODO: what does flag do ?
    pml_mpc_request_t* req = pml_mpc_request_of_ompi_req(request);
    mpc_lowcomm_request_cancel(&req->mpc_req);
    pml_mpc_verbose(30, "cancel_fn ompi_req %p", (void*)request);
    return 0;
}

/* // callback inside MPC */
/* int */
/* pml_mpc_request_completion_callback(mpc_lowcomm_request_t* mpc_reqp) */
/* { */
/*     pml_mpc_request_t* req = pml_mpc_request_of_mpc_req(mpc_reqp); */

/*     assert(req->ompi_req.req_type == OMPI_REQUEST_PML */
/*                 && req->ompi_req.req_state == OMPI_REQUEST_ACTIVE); */

/*     pml_mpc_verbose(30, */
/*                 "completion callback: completing mpc_req:%p pml_req:%p)", */
/*                 (void*)mpc_reqp, (void*)req); */
/*     ompi_request_complete(&req->ompi_req, true); */
/*     return 0; */
/* } */

// Pending list

void
pml_mpc_pending_list_init()
{
    OBJ_CONSTRUCT(&mca_pml_mpc_component.pending_list, opal_list_t);
}

void
pml_mpc_pending_list_fini()
{
    OPAL_LIST_DESTRUCT(&mca_pml_mpc_component.pending_list);
}

void
pml_mpc_pending_list_append(pml_mpc_request_t* req)
{
    opal_list_append(&mca_pml_mpc_component.pending_list, &req->pending_list_item);
}

int
pml_mpc_pending_list_size()
{
    return opal_list_get_size(&mca_pml_mpc_component.pending_list);
}

pml_mpc_request_t*
pml_mpc_pending_list_remove_first()
{
    opal_list_item_t* list_item
                = opal_list_remove_first(&mca_pml_mpc_component.pending_list);
    if (list_item == NULL) {
        return NULL;
    }
    return pml_mpc_request_of_pending_list_item(list_item);
}

// free-list
void
pml_mpc_requests_free_list_init()
{
    OBJ_CONSTRUCT(&mca_pml_mpc_component.requests_free_list, opal_free_list_t);
    opal_free_list_init(&mca_pml_mpc_component.requests_free_list, //&free_list,
                        sizeof(pml_mpc_request_t), opal_cache_line_size,
                        OBJ_CLASS(ompi_request_t), 0, opal_cache_line_size, 0, -1, 32,
                        NULL, 0, NULL, NULL, NULL);
}

void
pml_mpc_requests_free_list_fini()
{
    OBJ_DESTRUCT(&mca_pml_mpc_component.requests_free_list);
}

// adding status convertion in this file
void
pml_mpc_status_convert(mpc_lowcomm_status_t* mpc_status,
                       ompi_status_public_t* status)
{
    assert(mpc_status);
    if (status != NULL && status != MPI_STATUS_IGNORE) {
        status->MPI_SOURCE = mpc_status->MPC_SOURCE;
        status->MPI_TAG = mpc_status->MPC_TAG;
        status->MPI_ERROR = mpc_status->MPC_ERROR;
        status->_cancelled = mpc_status->cancelled;
        status->_ucount = mpc_status->size;
    }
}
