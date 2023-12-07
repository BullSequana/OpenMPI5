/*
 *
 * Copyright (c) 2006-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2007 The University of Tennessee and The University
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

#ifndef PML_MPC_H_HAS_BEEN_INCLUDED
#define PML_MPC_H_HAS_BEEN_INCLUDED

#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/bml/base/base.h"
#include "ompi/mca/pml/base/pml_base_bsend.h"
#include "ompi/mca/pml/base/pml_base_request.h"
#include "ompi/mca/pml/base/pml_base_sendreq.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/proc/proc.h"
#include "ompi/request/request.h"
#include "ompi_config.h"
#include "opal/class/opal_free_list.h"
#include "opal/mca/allocator/base/base.h"

BEGIN_C_DECLS

/* TODO: Find realistic values */
#define PML_MPC_MAX_CONTEXTID (0x7fffffff)
#define PML_MPC_MAX_TAG 65536

typedef struct mca_pml_mpc_component_t {
    /** Base pml component */
    mca_pml_base_component_2_0_0_t super;

    /** MCA parameter: Priority of this component */
    int priority;
    int verbose;
    /* 0 -> not implemented
     * 1  -> init
     * 5  -> communicator/proc add
     * 6  -> communicator init/fini
     * 10 -> isend / irecv
     * 20 -> completion
     * 30 -> request alloc/free etc
     * 30 -> request details init/fini etc
     * 40 -> temporary: send/recv
     * 50 -> progress polling
     */
    int output;
    int rank; // mpc rank for verbose output

    // free-list to allocate requests from
    opal_free_list_t requests_free_list;
    // Pending list of uncompleted requests
    opal_list_t pending_list;

} mca_pml_mpc_component_t;
extern mca_pml_mpc_component_t mca_pml_mpc_component;

typedef struct mca_pml_mpc_module_t {
    mca_pml_base_module_t super;

    /* global objects */

    /* requests */

    int priority;
    int verbose;
} mca_pml_mpc_module_t;
extern mca_pml_mpc_module_t mca_pml_mpc_module;

#define pml_mpc_verbose(lvl, fmt, ...)                                         \
    opal_output_verbose(lvl, mca_pml_mpc_component.output, "%d %s:%d " fmt,    \
                mca_pml_mpc_component.rank, __FILE__, __LINE__, ##__VA_ARGS__)
#define pml_mpc_error(fmt, ...)                                                \
    opal_output(0, "%s:%d ERROR: " fmt, __FILE__, __LINE__, ##__VA_ARGS__)

/*
 * PML interface functions.
 */
int
mca_pml_mpc_add_comm(struct ompi_communicator_t* comm);
int
mca_pml_mpc_del_comm(struct ompi_communicator_t* comm);

int
mca_pml_mpc_add_procs(struct ompi_proc_t** procs, size_t nprocs);
int
mca_pml_mpc_del_procs(struct ompi_proc_t** procs, size_t nprocs);

int
mca_pml_mpc_iprobe(int dst,
                   int tag,
                   struct ompi_communicator_t* comm,
                   int* matched,
                   ompi_status_public_t* status);

int
mca_pml_mpc_probe(int dst,
                  int tag,
                  struct ompi_communicator_t* comm,
                  ompi_status_public_t* status);

int
mca_pml_mpc_improbe(int dst,
                    int tag,
                    struct ompi_communicator_t* comm,
                    int* matched,
                    struct ompi_message_t** message,
                    ompi_status_public_t* status);

int
mca_pml_mpc_mprobe(int dst,
                   int tag,
                   struct ompi_communicator_t* comm,
                   struct ompi_message_t** message,
                   ompi_status_public_t* status);

int
mca_pml_mpc_cancel(ompi_request_t* request);
int
mca_pml_mpc_cancelled(ompi_request_t* request, int* flag);

int
mca_pml_mpc_isend_init(const void* buf,
                       size_t count,
                       ompi_datatype_t* datatype,
                       int dst,
                       int tag,
                       mca_pml_base_send_mode_t mode,
                       struct ompi_communicator_t* comm,
                       struct ompi_request_t** request);

int
mca_pml_mpc_isend(const void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  int dst,
                  int tag,
                  mca_pml_base_send_mode_t mode,
                  struct ompi_communicator_t* comm,
                  struct ompi_request_t** request);

int
mca_pml_mpc_send(const void* buf,
                 size_t count,
                 ompi_datatype_t* datatype,
                 int dst,
                 int tag,
                 mca_pml_base_send_mode_t mode,
                 struct ompi_communicator_t* comm);

int
mca_pml_mpc_irecv_init(void* buf,
                       size_t count,
                       ompi_datatype_t* datatype,
                       int src,
                       int tag,
                       struct ompi_communicator_t* comm,
                       struct ompi_request_t** request);

int
mca_pml_mpc_irecv(void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  int src,
                  int tag,
                  struct ompi_communicator_t* comm,
                  struct ompi_request_t** request);

int
mca_pml_mpc_recv(void* buf,
                 size_t count,
                 ompi_datatype_t* datatype,
                 int src,
                 int tag,
                 struct ompi_communicator_t* comm,
                 ompi_status_public_t* status);

int
mca_pml_mpc_imrecv(void* buf,
                   size_t count,
                   ompi_datatype_t* datatype,
                   struct ompi_message_t** message,
                   struct ompi_request_t** request);

int
mca_pml_mpc_mrecv(void* buf,
                  size_t count,
                  ompi_datatype_t* datatype,
                  struct ompi_message_t** message,
                  ompi_status_public_t* status);

int
mca_pml_mpc_progress(void);

int
mca_pml_mpc_start(size_t count, ompi_request_t** requests);

int
mca_pml_mpc_ft_event(int state);

int
mca_pml_mpc_enable(bool enable);

int
mca_pml_mpc_dump(struct ompi_communicator_t* comm, int verbose);

END_C_DECLS

#endif /* PML_MPC_H_HAS_BEEN_INCLUDED */
