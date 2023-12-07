/*
 * Copyright (c) 2023-2024 BULL S.A.S. All rights reserved.
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
 * Experimental
 * Basic scatterw
 */

int
mca_coll_basic_scatterw_intra(const void *sbuf, const int *scounts, const int *disps,
                              struct ompi_datatype_t **sdtypes,
                              void *rbuf, int rcount,
                              struct ompi_datatype_t *rdtype,
                              int root,
                              struct ompi_communicator_t *comm,
                              const mca_coll_base_module_t *module)
{
    int rank;
    int size;
    int err = MPI_SUCCESS;

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    /* Recv data from root */
    if (rank != root) {
        if (rcount > 0) {
            int ret = MCA_PML_CALL(recv(rbuf,
                                        rcount,
                                        rdtype,
                                        root,
                                        MCA_COLL_BASE_TAG_SCATTERW,
                                        comm,
                                        MPI_STATUS_IGNORE));
            return ret;
        }
        return MPI_SUCCESS;
    }

    int nb_req = 0;
    ompi_request_t **request = NULL;
    request = ompi_coll_base_comm_get_reqs(module->base_data, size);
    if (NULL == request) { return OMPI_ERR_OUT_OF_RESOURCE; }

    for (int i = 0; i < size; ++i) {
        const char *tmp = ((const char *) sbuf) + disps[i];
        if (i == rank) {
            /* Self send */
            if (MPI_IN_PLACE != rbuf && (0 < rcount) && (0 < scounts[i])) {
                err = ompi_datatype_sndrcv(tmp,
                                           scounts[i],
                                           sdtypes[i],
                                           rbuf,
                                           rcount,
                                           rdtype);
            }
        } else {
            /* Only send if there is something to be sent */
            if (scounts[i] > 0) {
                err = MCA_PML_CALL(isend(tmp,
                                         scounts[i],
                                         sdtypes[i],
                                         i,
                                         MCA_COLL_BASE_TAG_SCATTERW,
                                         MCA_PML_BASE_SEND_STANDARD,
                                         comm,
                                         &request[nb_req]));
                nb_req++;
            }
        }
        if (MPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(request, nb_req);
            return err;
        }
    }


    err = ompi_request_wait_all(nb_req, request, MPI_STATUSES_IGNORE);
    if (MPI_SUCCESS != err) {
        ompi_coll_base_free_reqs(request, nb_req);
    }

    /* All done */
    return err;
}

