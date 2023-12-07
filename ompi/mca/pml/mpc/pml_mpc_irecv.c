/*
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2011      Sandia National Laboratories. All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "mpc_lowcomm.h"
#include "ompi/request/request.h"
#include "ompi_config.h"
#include "pml_mpc.h"
#include "pml_mpc_communicator.h"
#include "pml_mpc_pack.h"
#include "pml_mpc_request.h"

int
mca_pml_mpc_recv(void* buf,
                 size_t count,
                 ompi_datatype_t* datatype,
                 int src,
                 int tag,
                 struct ompi_communicator_t* comm,
                 ompi_status_public_t* status)
{
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);

    char* recv_buf = buf;
    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        recv_buf = malloc(datatype_size * count);
    }

    int mpc_src = MPI_ANY_SOURCE == src ? MPC_ANY_SOURCE
                                        : pml_mpc_comm_rank(comm, src);
    int mpc_tag = MPI_ANY_TAG == tag ? MPC_ANY_TAG : tag;
    pml_mpc_verbose(10, "recv(%d [%d], size=%ld)", src, mpc_src,
                    datatype_size * count);
    mpc_lowcomm_request_t mpc_req;
    mpc_lowcomm_irecv(mpc_src, recv_buf, datatype_size * count, mpc_tag,
                      pml_mpc_comm(comm), &mpc_req);
    mpc_lowcomm_status_t mpc_status;
    mpc_lowcomm_wait(&mpc_req, &mpc_status);
    pml_mpc_status_convert(&mpc_status, status);

    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        int ret = mca_pml_mpc_unpack(buf, recv_buf, count, datatype);
        free(recv_buf);
        return ret;
    }
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_irecv(void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  int src,
                  int tag,
                  struct ompi_communicator_t* comm,
                  struct ompi_request_t** request)
{
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);

    pml_mpc_request_t* req = pml_mpc_request_alloc();
    pml_mpc_request_init(req, false);
    void* recv_buf = buf;
    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        req->irecv_needs_unpack = true;
        req->tmp_buf = malloc(datatype_size * count);
        recv_buf = req->tmp_buf;
        req->has_args = PML_MPC_IRECV_ARGS;
        req->args.irecv = (pml_mpc_irecv_args_t){
            .buf = buf,
            .count = count,
            .datatype = datatype,
            .src = src,
            .tag = tag,
            .comm = comm,
        };
    }
    int mpc_src = MPI_ANY_SOURCE == src ? MPC_ANY_SOURCE
                                        : pml_mpc_comm_rank(comm, src);
    int mpc_tag = MPI_ANY_TAG == tag ? MPC_ANY_TAG : tag;
    mpc_lowcomm_irecv(mpc_src, recv_buf, datatype_size * count, mpc_tag,
                pml_mpc_comm(comm), &req->mpc_req);
    if (req->mpc_req.completion_flag) {
        // immediate
        ompi_request_complete(&req->ompi_req, true);
    } else {
        pml_mpc_pending_list_append(req);
    }

    pml_mpc_verbose(10,
                    "irecv(%d [%d], size=%ld) : req %p mpc_req %p mpc completion %s",
                    src, mpc_src, datatype_size * count, (void*)req,
                    (void*)&req->mpc_req,
                    req->ompi_req.req_complete ? "immediate" : "defered");
    *request = &req->ompi_req;
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_irecv_init(void* buf,
                       size_t count,
                       ompi_datatype_t* datatype,
                       int src,
                       int tag,
                       struct ompi_communicator_t* comm,
                       struct ompi_request_t** request)
{
    pml_mpc_request_t* req = pml_mpc_request_alloc();
    pml_mpc_request_init(req, true);
    req->ompi_req.req_state = OMPI_REQUEST_INACTIVE;
    req->has_args = PML_MPC_IRECV_ARGS;
    req->args.irecv = (pml_mpc_irecv_args_t){
        .buf = buf,
        .count= count,
        .datatype = datatype,
        .src = src,
        .tag = tag,
        .comm = comm,
    };
    if (!ompi_datatype_is_contiguous_memory_layout(datatype, count)) {
        size_t datatype_size;
        ompi_datatype_type_size(datatype, &datatype_size);
        req->tmp_buf = malloc(count * datatype_size);
        req->irecv_needs_unpack = true;
    }

    *request = &req->ompi_req;
    return OMPI_SUCCESS;
}

// irecv, but with persistant request
int
mca_pml_mpc_irecv_start(pml_mpc_request_t* req)
{
    if (req->has_args != PML_MPC_IRECV_ARGS) {
        pml_mpc_error("bad request %p", (void*)req);
    }
    pml_mpc_irecv_args_t* args = &req->args.irecv;
    size_t count = args->count;
    ompi_datatype_t* datatype = args->datatype;
    int src = args->src;
    int tag = args->tag;
    ompi_communicator_t* comm = args->comm;

    void* recv_buf;
    if (NULL != req->tmp_buf) { // non contiguous case: buffering
        recv_buf = req->tmp_buf;
    } else {
        recv_buf = args->buf;
    }
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);

    int mpc_src = MPI_ANY_SOURCE == src ? MPC_ANY_SOURCE
                                        : pml_mpc_comm_rank(comm, src);
    int mpc_tag = MPI_ANY_TAG == tag ? MPC_ANY_TAG : tag;
    mpc_lowcomm_irecv(mpc_src, recv_buf, datatype_size * count, mpc_tag,
                      pml_mpc_comm(comm), &req->mpc_req);
    if (req->mpc_req.completion_flag) {
        // immediate
        ompi_request_complete(&req->ompi_req, true);
    } else {
        pml_mpc_pending_list_append(req);
    }

    pml_mpc_verbose(10, "irecv_start(%d [%d], size=%ld) : req %p mpc_req %p mpc "
                    "completion %s",
                    src, mpc_src, datatype_size * count, (void*)req,
                    (void*)&req->mpc_req,
                    req->ompi_req.req_complete ? "immediate" : "defered");
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_imrecv(void* buf,
                   size_t count,
                   ompi_datatype_t* datatype,
                   struct ompi_message_t** message,
                   struct ompi_request_t** request)
{
    (void)buf;
    (void)count;
    (void)datatype;
    (void)message;
    (void)request;
    pml_mpc_error("imcrecv() not implemented");
    return OMPI_ERROR;
}

int
mca_pml_mpc_mrecv(void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  struct ompi_message_t** message,
                  ompi_status_public_t* status)
{
    (void)buf;
    (void)count;
    (void)datatype;
    (void)message;
    (void)status;
    pml_mpc_error("mrecv() not implemented");
    return OMPI_ERROR;
}
