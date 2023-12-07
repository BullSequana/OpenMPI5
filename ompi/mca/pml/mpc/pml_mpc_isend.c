/*
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "mpc_lowcomm.h"
#include "ompi_config.h"
#include "pml_mpc.h"
#include "pml_mpc_communicator.h"
#include "pml_mpc_pack.h"
#include "pml_mpc_request.h"

int
mca_pml_mpc_init_send_buffered(const void* buf,
                               size_t count,
                               ompi_datatype_t* datatype,
                               mca_pml_base_send_mode_t mode,
                               pml_mpc_request_t* req);

int
mca_pml_mpc_send(const void* buf,
                 size_t count,
                 ompi_datatype_t* datatype,
                 int dst,
                 int tag,
                 mca_pml_base_send_mode_t mode,
                 ompi_communicator_t* comm)
{
    if (mode == MCA_PML_BASE_SEND_BUFFERED) {
        struct ompi_request_t *req = NULL;
        return mca_pml_mpc_isend(buf, count, datatype,
                                 dst, tag, mode, comm, &req);
    }
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    int mpc_dst = pml_mpc_comm_rank(comm, dst);

    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        void* tmp_buf = malloc(datatype_size * count);
        int ret = mca_pml_mpc_pack(tmp_buf, buf, count, datatype);
        if (OMPI_SUCCESS != ret) {
            free(tmp_buf);
            return OMPI_ERROR;
        }
        pml_mpc_verbose(10, "send(%d [%d], packed size=%ld)", dst, mpc_dst,
                        datatype_size * count);
        mpc_lowcomm_send(mpc_dst, tmp_buf, datatype_size * count, tag,
                         pml_mpc_comm(comm));
        free(tmp_buf);
        return OMPI_SUCCESS;
    }

    pml_mpc_verbose(10, "send(%d [%d], size=%ld)", dst, mpc_dst,
                    datatype_size * count);
    mpc_lowcomm_send(mpc_dst, buf, datatype_size * count, tag, pml_mpc_comm(comm));
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_isend(const void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  int dst,
                  int tag,
                  mca_pml_base_send_mode_t mode,
                  ompi_communicator_t* comm,
                  ompi_request_t** request)
{
    const void* send_buf = NULL;
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    pml_mpc_request_t* req = pml_mpc_request_alloc();
    pml_mpc_request_init(req, false);
    if (mode == MCA_PML_BASE_SEND_BUFFERED || !ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
       int ret = mca_pml_mpc_init_send_buffered(buf, count, datatype, mode, req);
       if (ret != OMPI_SUCCESS) {
           return ret;
       }
       send_buf = req->tmp_buf;
    } else {
       send_buf = buf;
    }

    int mpc_dst = pml_mpc_comm_rank(comm, dst);
    mpc_lowcomm_isend(mpc_dst, send_buf, datatype_size * count, tag,
                      pml_mpc_comm(comm), &req->mpc_req);
    if (req->mpc_req.completion_flag) {
        // immediate
        ompi_request_complete(&req->ompi_req, true);
    } else {
        pml_mpc_pending_list_append(req);
    }

    pml_mpc_verbose(10, "isend(%d [%d], size=%ld, mpccom=%lu)"
                    " : req %p mpc_req %p mpc completion %s",
                    dst, mpc_dst, datatype_size * count,
                    mpc_lowcomm_communicator_id(pml_mpc_comm(comm)), (void*)req,
                    (void*)&req->mpc_req,
                    req->ompi_req.req_complete ? "immediate" : "defered");
    *request = &req->ompi_req;
    return OMPI_SUCCESS;
}

//Function use when a buffer is needed for a send
int
mca_pml_mpc_init_send_buffered(const void* buf,
                               size_t count,
                               ompi_datatype_t* datatype,
                               mca_pml_base_send_mode_t mode,
                               pml_mpc_request_t* req)
{

    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    if (mode == MCA_PML_BASE_SEND_BUFFERED) {
        size_t msg_size;
        msg_size = datatype_size * count;
        req->tmp_buf = mca_pml_base_bsend_request_alloc_buf(msg_size);
        if(req->tmp_buf == NULL) {
            pml_mpc_error("Attached buffer too small !");
            pml_mpc_request_fini(req);
            return OMPI_ERROR;
        }
        req->buffered = true;
    } else {
        req->tmp_buf = malloc(datatype_size * count);
    }
    int ret = mca_pml_mpc_pack(req->tmp_buf, buf, count, datatype);
    if (OMPI_SUCCESS != ret) {
        pml_mpc_request_fini(req);
        return OMPI_ERROR;
    }
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_isend_init(const void* buf,
                       size_t count,
                       ompi_datatype_t* datatype,
                       int dst,
                       int tag,
                       mca_pml_base_send_mode_t mode,
                       ompi_communicator_t* comm,
                       ompi_request_t** request)
{
    pml_mpc_request_t* req = pml_mpc_request_alloc();
    pml_mpc_request_init(req, true);
    req->ompi_req.req_start = mca_pml_mpc_start;
    req->ompi_req.req_state = OMPI_REQUEST_INACTIVE;
    req->has_args = PML_MPC_ISEND_ARGS;
    req->args.isend = (pml_mpc_isend_args_t){
        .buf = buf,
        .count = count,
        .datatype = datatype,
        .dst = dst,
        .tag = tag,
        .mode = mode,
        .comm = comm,
    };

    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        size_t datatype_size;
        ompi_datatype_type_size(datatype, &datatype_size);
        req->tmp_buf = malloc(datatype_size * count);
    }
    *request = &req->ompi_req;
    return OMPI_SUCCESS;
}

// isend with persistant request
int
mca_pml_mpc_isend_start(pml_mpc_request_t* req)
{
    if (req->has_args != PML_MPC_ISEND_ARGS) {
        pml_mpc_error("bad request %p", (void*)req);
    }
    pml_mpc_isend_args_t* args = &req->args.isend;
    const void* buf = args->buf;
    size_t count = args->count;
    ompi_datatype_t* datatype = args->datatype;
    int dst = args->dst;
    int tag = args->tag;
    mca_pml_base_send_mode_t mode = args->mode;
    ompi_communicator_t* comm = args->comm;
    const void* send_buf = NULL;
    if (mode == MCA_PML_BASE_SEND_BUFFERED || !ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
       int ret = mca_pml_mpc_init_send_buffered(buf, count, datatype, mode, req);
       if (ret != OMPI_SUCCESS) {
           return ret;
       }
       send_buf = req->tmp_buf;
    } else {
       send_buf = buf;
    }
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    int mpc_dst = pml_mpc_comm_rank(comm, dst);
    mpc_lowcomm_isend(mpc_dst, send_buf, datatype_size * count, tag,
                      pml_mpc_comm(comm), &req->mpc_req);
    if (req->mpc_req.completion_flag) {
        // immediate
        ompi_request_complete(&req->ompi_req, true);
    } else {
        pml_mpc_pending_list_append(req);
    }

    pml_mpc_verbose(10, "isend_start(%d [%d], size=%ld) : req %p mpc_req %p mpc "
                    "completion %s",
                    dst, mpc_dst, datatype_size * count, (void*)req,
                    (void*)&req->mpc_req,
                    req->ompi_req.req_complete ? "immediate" : "defered");
    return OMPI_SUCCESS;
}
