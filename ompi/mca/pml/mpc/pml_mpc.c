/*
 * Copyright (c) 2006-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
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
#include "pml_mpc.h"
#include "mpc_lowcomm.h"
#include "pml_mpc_communicator.h"
#include "pml_mpc_request.h"

mca_pml_mpc_module_t mca_pml_mpc_module
            = { .super = { .pml_add_procs = mca_pml_mpc_add_procs,
                            .pml_del_procs = mca_pml_mpc_del_procs,
                            .pml_enable = mca_pml_mpc_enable,
                            .pml_progress = mca_pml_mpc_progress,
                            .pml_add_comm = mca_pml_mpc_add_comm,
                            .pml_del_comm = mca_pml_mpc_del_comm,
                            .pml_irecv_init = mca_pml_mpc_irecv_init,
                            .pml_irecv = mca_pml_mpc_irecv,
                            .pml_recv = mca_pml_mpc_recv,
                            .pml_isend_init = mca_pml_mpc_isend_init,
                            .pml_isend = mca_pml_mpc_isend,
                            .pml_send = mca_pml_mpc_send,
                            .pml_iprobe = mca_pml_mpc_iprobe,
                            .pml_probe = mca_pml_mpc_probe,
                            .pml_start = mca_pml_mpc_start,
                            .pml_improbe = mca_pml_mpc_improbe,
                            .pml_mprobe = mca_pml_mpc_mprobe,
                            .pml_imrecv = mca_pml_mpc_imrecv,
                            .pml_mrecv = mca_pml_mpc_mrecv,
                            .pml_dump = mca_pml_mpc_dump,
                            .pml_ft_event = mca_pml_mpc_ft_event,
                            .pml_max_contextid = PML_MPC_MAX_CONTEXTID,
                            .pml_max_tag = PML_MPC_MAX_TAG } };

int
mca_pml_mpc_add_comm(struct ompi_communicator_t* comm)
{
    pml_mpc_verbose(5, "add_comm() cid=%d (%s) csize=%d crank=%d",
                    comm->c_contextid, comm->c_name, ompi_comm_size(comm),
                    ompi_comm_rank(comm));
    return pml_mpc_communicator_init(comm);
}

int
mca_pml_mpc_del_comm(struct ompi_communicator_t* comm)
{
    return pml_mpc_communicator_fini(comm);
}

int
mca_pml_mpc_add_procs(struct ompi_proc_t** procs, size_t nprocs)
{
    (void)procs;
    (void)nprocs;
    pml_mpc_verbose(5, "dummy add_procs()");
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_del_procs(struct ompi_proc_t** procs, size_t nprocs)
{
    (void)procs;
    (void)nprocs;
    pml_mpc_verbose(5, "dummy del_procs()");
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_enable(bool enable)
{
    if (false == enable) {
        return OMPI_SUCCESS;
    }

    /* else TODO */

    /* mca_pml_mpc.enabled = true; */
    return OMPI_SUCCESS;
}

// TODO: what should this function do ?
int
mca_pml_mpc_dump(struct ompi_communicator_t* comm, int verbose)
{
    (void)comm;
    (void)verbose;
    pml_mpc_error("dump() is not implemented yet\n");
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_ft_event(int state)
{
    (void)state;
    pml_mpc_error("ft_event() is not implemented yet\n");
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_progress(void)
{
    int pending_num = pml_mpc_pending_list_size();
    int i;
    for (i = 0; i < pending_num; i++) {
        pml_mpc_request_t* req = pml_mpc_pending_list_remove_first();
        if (NULL == req) {
            break;
        }
        int completed = 0;
        mpc_lowcomm_test(&req->mpc_req, &completed, NULL);
        if (completed) {
            // completion callback should complete the request,
            // meanwhile completing manually here
            ompi_request_complete(&req->ompi_req, true);
            pml_mpc_verbose(20,
                            "progress() completed req %p mpc_req: %p pending=%d",
                            (void*)req, (void*)&req->mpc_req,
                            pml_mpc_pending_list_size());
        } else {
            pml_mpc_pending_list_append(req); // append at the end
            pml_mpc_verbose(50, "progress() test(req=%p) pending=%d",
                            (void*)req, pml_mpc_pending_list_size());
        }
    }
    if (0 == pending_num) {
        pml_mpc_verbose(51, "progress()");
    }
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_iprobe(int dst,
                   int tag,
                   struct ompi_communicator_t* comm,
                   int* matched,
                   ompi_status_public_t* status)
{
    pml_mpc_verbose(20, "iprobe()");
    mpc_lowcomm_status_t mpc_status;
    int mpc_dst = MPI_ANY_SOURCE == dst ? MPC_ANY_SOURCE
                                        : pml_mpc_comm_rank(comm, dst);
    int mpc_tag = MPI_ANY_TAG == tag ? MPC_ANY_TAG : tag;
    mpc_lowcomm_iprobe(mpc_dst, mpc_tag, pml_mpc_comm(comm), matched, &mpc_status);
    pml_mpc_status_convert(&mpc_status, status);
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_probe(int dst,
                  int tag,
                  struct ompi_communicator_t* comm,
                  ompi_status_public_t* status)
{
    pml_mpc_verbose(20, "probe()");
    mpc_lowcomm_status_t mpc_status;
    int mpc_dst = MPI_ANY_SOURCE == dst ? MPC_ANY_SOURCE
                                        : pml_mpc_comm_rank(comm, dst);
    int mpc_tag = MPI_ANY_TAG == tag ? MPC_ANY_TAG : tag;
    mpc_lowcomm_probe(mpc_dst, mpc_tag, pml_mpc_comm(comm), &mpc_status);
    pml_mpc_status_convert(&mpc_status, status);
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_start(size_t count, ompi_request_t** requests)
{
    size_t i = 0;
    for (i = 0; i < count; i++) {
        pml_mpc_request_t* req = pml_mpc_request_of_ompi_req(requests[i]);
        req->ompi_req.req_state = OMPI_REQUEST_ACTIVE;
        switch (req->has_args) {
            case PML_MPC_ISEND_ARGS:
                if(mca_pml_mpc_isend_start(req) == OMPI_ERROR){
                    return OMPI_ERROR;
                }
                break;
            case PML_MPC_IRECV_ARGS:
                if(mca_pml_mpc_irecv_start(req) == OMPI_ERROR){
                    return OMPI_ERROR;
                }
                break;
            case PML_MPC_NO_ARGS:
                return OMPI_ERROR;
        }
    }
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_improbe(int dst,
                    int tag,
                    struct ompi_communicator_t* comm,
                    int* matched,
                    struct ompi_message_t** message,
                    ompi_status_public_t* status)
{
    (void)dst;
    (void)tag;
    (void)comm;
    (void)matched;
    (void)message;
    (void)status;
    pml_mpc_error("improbe() not implemented");
    return OMPI_ERROR;
}

int
mca_pml_mpc_mprobe(int dst,
                   int tag,
                   struct ompi_communicator_t* comm,
                   struct ompi_message_t** message,
                   ompi_status_public_t* status)
{
    (void)dst;
    (void)tag;
    (void)comm;
    (void)message;
    (void)status;
    pml_mpc_error("mprobe() not implemented");
    return OMPI_ERROR;
}
