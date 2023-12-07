/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi/communicator/communicator.h"

#include "mpc_lowcomm.h"
#include "pml_mpc.h"
#include "pml_mpc_communicator.h"
#include "mpc_lowcomm_group.h"
#include "mpc_lowcomm_communicator.h"
#include <limits.h>
#include <math.h>

// utility function to print an int array, used for verbose/debug
static char*
pml_mpc_sprint_intarray(int n, int* arr)
{
    int int_digits = ceil(log10((double)INT_MAX)) + 1;  // + sign
    int bufsize = (int_digits + 1 /* space */) * n + 1; // trailing '\0'
    char* buf = malloc(bufsize);
    if (NULL == buf) {
        return NULL;
    }
    int written = 0;
    int i;
    for (i = 0; i < n; i++) {
        written += snprintf(buf + written, bufsize - written, "%d ", arr[i]);
    }
    return buf;
}

// ranks of a communicator as world ranks
int*
pml_mpc_as_world_ranks(ompi_communicator_t* ompi_comm)
{
    int n = ompi_comm_size(ompi_comm);
    ompi_group_t* group;
    ompi_comm_group(ompi_comm, &group);
    ompi_group_t* world_group;
    ompi_comm_group(MPI_COMM_WORLD, &world_group);

    int* ranks = malloc(n * sizeof(int));
    if (NULL == ranks) {
        return NULL;
    }
    int i;
    for (i = 0; i < n; i++) {
        ranks[i] = i;
    }
    int* world_ranks = malloc(n * sizeof(int));
    if (NULL == world_ranks) {
        free(ranks);
        return NULL;
    }
    // translate all ranks [0, n] in terms of COMM_WORLD_ranks
    ompi_group_translate_ranks(group, n, ranks, world_group, world_ranks);

    free(ranks);
    return world_ranks;
}

int
pml_mpc_communicator_init(ompi_communicator_t* ompi_comm)
{
    if (NULL == ompi_comm) {
        return OMPI_ERROR;
    }

    int nb_tasks = ompi_comm_size(ompi_comm);
    mpc_lowcomm_communicator_t mpc_comm = MPC_COMM_NULL;
    if (MPI_COMM_NULL == ompi_comm) {
        mpc_comm = MPC_COMM_NULL;
    } else if (MPI_COMM_WORLD == ompi_comm) {
        mpc_comm = MPC_COMM_WORLD;
    } else if (MPI_COMM_SELF == ompi_comm) {
        mpc_comm = MPC_COMM_SELF;
    } else {
        int* ranks_as_world_ranks = pml_mpc_as_world_ranks(ompi_comm);
        if (NULL == ranks_as_world_ranks) {
            pml_mpc_error("unable to allocate rank translation");
            return OMPI_ERROR;
        }
        /*
         * create MPC communicator:
         *
         * - the MPC function is not a collective on MPC_COMM_WORLD:
         *   only selected ranks participate so it does not hang
         *
         * - The new MPC communicator is only used as a way to have a
         *   new isolated context for MPC send/recv operations.
         *
         * - The PML API only registers add_comm() and does not do
         *   much with respect to communicators. So the PML/MPC is not
         *   a wrapper of MPC communicators and most operations on
         *   communicators/groups are done elsewhere in OMPI
         *   communicator/group code.
         *
         * - The new MPC communicator does not have attributes or
         *   parent set. In fact at this stage in ompi_communicator_t
         *   we do not have access to the parent OMPI communicator.
         */
        mpc_lowcomm_group_t* mpc_group = mpc_lowcomm_group_create(nb_tasks, ranks_as_world_ranks);
        mpc_lowcomm_communicator_create_group(MPC_COMM_WORLD, mpc_group, 0, &mpc_comm);
        if (MPC_COMM_NULL == mpc_comm) {
            pml_mpc_error("MPC_COMM_NULL returned creating mpc comm from ompi_cid=%d ",
                          ompi_comm->c_contextid);
            free(ranks_as_world_ranks);
            return OMPI_ERROR;
        }
        if (mca_pml_mpc_component.verbose >= 6) {
            char* buf = pml_mpc_sprint_intarray(nb_tasks, ranks_as_world_ranks);
            pml_mpc_verbose(6, "communicator_create: ompi_id=%d nb_tasks=%d [%s] "
                            "-> %p mpc_id=%lu", ompi_comm->c_contextid, nb_tasks, buf, mpc_comm,
                            mpc_lowcomm_communicator_id(mpc_comm));
            free(buf);
        }
        free(ranks_as_world_ranks);
    }

    mca_pml_comm_t* pml_comm = malloc(sizeof(*pml_comm));
    if (NULL == pml_comm) {
        pml_mpc_error("unable to allocate pml communicator");
        return OMPI_ERROR;
    }
    pml_comm->mpc_comm = mpc_comm;

    // register as PML data:
    ompi_comm->c_pml_comm = (struct mca_pml_comm_t*)pml_comm;

    return OMPI_SUCCESS;
}

int
pml_mpc_communicator_fini(ompi_communicator_t* ompi_comm)
{
    if (NULL == ompi_comm) {
        return OMPI_ERROR;
    }
    mca_pml_comm_t* pml_comm
                = (mca_pml_comm_t*)ompi_comm->c_pml_comm;
    ompi_comm->c_pml_comm = NULL;

    if (NULL != pml_comm) {
        mpc_lowcomm_communicator_t mpc_comm = pml_comm->mpc_comm;
        free(pml_comm);
        int free_mpc_comm = (NULL != mpc_comm && MPC_COMM_NULL != mpc_comm
                             && MPC_COMM_WORLD != mpc_comm
                             && MPC_COMM_SELF != mpc_comm);

        pml_mpc_verbose(6, "communicator_fini: ompi_comm_cid=%d mpc_comm=%p id=%lu"
                        " count=%d free_mpc_comm=%d",
                        ompi_comm->c_contextid, mpc_comm,
                        mpc_lowcomm_communicator_id(mpc_comm),
                        mpc_lowcomm_communicator_get_process_count(mpc_comm),
                        free_mpc_comm);
        if (free_mpc_comm) {
            if (MPC_LOWCOMM_SUCCESS
                        != mpc_lowcomm_communicator_free(&mpc_comm)) {
                return OMPI_ERROR;
            }
        }
    }
    return OMPI_SUCCESS;
}

// MPC communictor corresponding to OMPI communicator
mpc_lowcomm_communicator_t
pml_mpc_comm(ompi_communicator_t* ompi_comm)
{
    assert(ompi_comm);
    if (MPI_COMM_NULL == ompi_comm) {
        return MPC_COMM_NULL;
    } else if (MPI_COMM_WORLD == ompi_comm) {
        return MPC_COMM_WORLD;
    } else if (MPI_COMM_SELF == ompi_comm) {
        return MPC_COMM_SELF;
    }
    mca_pml_comm_t* pml_comm
                = (mca_pml_comm_t*)ompi_comm->c_pml_comm;
    assert(pml_comm);
    return pml_comm->mpc_comm;
}

// mpc rank
// Using MPC communicators so don't need to translate
int
pml_mpc_comm_rank(ompi_communicator_t* ompi_comm, int rank)
{
    (void)ompi_comm; // unused
    return rank;
}
