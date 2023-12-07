/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2016 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_basic.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

/**
 *  * Experimental
 *   * Basic gatherw
 *    */
int
mca_coll_basic_gatherw_intra(const void *sbuf, int scount,
                             struct ompi_datatype_t *sdtype,
                             void *rbuf, const int *rcounts,
                             const int *disps,
                             struct ompi_datatype_t **rdtypes,
                             int root,
                             struct ompi_communicator_t *comm,
                             mca_coll_base_module_t *module)
{
    int i, rank, size, err;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* Send data to root */
    if (rank != root) {
        if (scount > 0) {
            return MCA_PML_CALL(send(sbuf,
                                     scount,
                                     sdtype,
                                     root,
                                     MCA_COLL_BASE_TAG_GATHERW,
                                     MCA_PML_BASE_SEND_STANDARD,
                                     comm));
        }
        return MPI_SUCCESS;
    }

    for (i = 0; i < size; ++i) {
        char *ptmp = ((char *) rbuf) + disps[i];

        if (i == rank) {
            /* Self send */
            if (MPI_IN_PLACE != sbuf && (0 < scount) && (0 < rcounts[i])) {
                err = ompi_datatype_sndrcv(sbuf,
                                           scount,
                                           sdtype,
                                           ptmp,
                                           rcounts[i],
                                           rdtypes[i]);
            }
        } else {
            /* Only receive if there is something to receive */
            if (rcounts[i] > 0) {
                err = MCA_PML_CALL(recv(ptmp,
                                        rcounts[i],
                                        rdtypes[i],
                                        i,
                                        MCA_COLL_BASE_TAG_GATHERW,
                                        comm,
                                        MPI_STATUS_IGNORE));
            }
        }
        if (MPI_SUCCESS != err) {
            return err;
        }
    }

    /* All done */

    return MPI_SUCCESS;
}

