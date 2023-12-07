/*
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef PML_MPC_COMMUNICATOR_H
#define PML_MPC_COMMUNICATOR_H
#include "ompi/communicator/communicator.h"

#include "mpc_lowcomm.h"
#include "pml_mpc.h"

typedef struct mca_pml_comm_t {
    mpc_lowcomm_communicator_t mpc_comm;
} mca_pml_comm_t;

int
pml_mpc_communicator_init(ompi_communicator_t* ompi_comm);
int
pml_mpc_communicator_fini(ompi_communicator_t* ompi_comm);

// MPC communictor corresponding to OMPI communicator
mpc_lowcomm_communicator_t
pml_mpc_create_mpc_comm(ompi_communicator_t* ompi_comm);
// retrieve it
mpc_lowcomm_communicator_t
pml_mpc_comm(ompi_communicator_t* ompi_comm);

int
pml_mpc_comm_rank(ompi_communicator_t* ompi_comm, int rank);

// ranks of a communicator as world ranks
int*
pml_mpc_as_world_ranks(ompi_communicator_t* ompi_comm);
#endif
