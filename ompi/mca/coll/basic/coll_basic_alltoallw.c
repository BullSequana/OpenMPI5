/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2021 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2012      Oak Ridge National Labs.  All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2013      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2014-2021 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2014      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2017      IBM Corporation. All rights reserved.
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
#include "opal/datatype/opal_convertor_internal.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"

/*
 *	alltoallw_inter
 *
 *	Function:	- MPI_Alltoallw
 *	Accepts:	- same as MPI_Alltoallw()
 *	Returns:	- MPI_SUCCESS or an MPI error code
 */
int
mca_coll_basic_alltoallw_inter(const void *sbuf, const int *scounts, const int *sdisps,
                               struct ompi_datatype_t * const *sdtypes,
                               void *rbuf, const int *rcounts, const int *rdisps,
                               struct ompi_datatype_t * const *rdtypes,
                               struct ompi_communicator_t *comm,
                               mca_coll_base_module_t *module)
{
    int i, size, err, nreqs;
    char *psnd, *prcv;
    ompi_request_t **preq, **reqs;

    /* Initialize. */
    size = ompi_comm_remote_size(comm);

    /* Initiate all send/recv to/from others. */
    nreqs = 0;
    reqs = preq = ompi_coll_base_comm_get_reqs(module->base_data, 2 * size);
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* Post all receives first -- a simple optimization */
    for (i = 0; i < size; ++i) {
        size_t msg_size;
        ompi_datatype_type_size(rdtypes[i], &msg_size);
        msg_size *= rcounts[i];

        if (0 == msg_size)
            continue;

        prcv = ((char *) rbuf) + rdisps[i];
        err = MCA_PML_CALL(irecv_init(prcv, rcounts[i], rdtypes[i],
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      comm, preq++));
        ++nreqs;
        if (OMPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Now post all sends */
    for (i = 0; i < size; ++i) {
        size_t msg_size;
        ompi_datatype_type_size(sdtypes[i], &msg_size);
        msg_size *= scounts[i];

        if (0 == msg_size)
            continue;

        psnd = ((char *) sbuf) + sdisps[i];
        err = MCA_PML_CALL(isend_init(psnd, scounts[i], sdtypes[i],
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      MCA_PML_BASE_SEND_STANDARD, comm,
                                      preq++));
        ++nreqs;
        if (OMPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Start your engines.  This will never return an error. */
    MCA_PML_CALL(start(nreqs, reqs));

    /* Wait for them all.  If there's an error, note that we don't care
     * what the error was -- just that there *was* an error.  The PML
     * will finish all requests, even if one or more of them fail.
     * i.e., by the end of this call, all the requests are free-able.
     * So free them anyway -- even if there was an error, and return the
     * error after we free everything. */
    err = ompi_request_wait_all(nreqs, reqs, MPI_STATUSES_IGNORE);

    /* Free the requests in all cases as they are persistent */
    ompi_coll_base_free_reqs(reqs, nreqs);

    /* All done */
    return err;
}
