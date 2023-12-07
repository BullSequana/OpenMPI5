/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2023      BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "mpi.h"
#include "ompi/constants.h"
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_convertor_internal.h"
#include "ompi/communicator/communicator.h"
#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "coll_base_topo.h"
#include "coll_base_util.h"

/*
 * We want to minimize the amount of temporary memory needed while allowing as many ranks
 * to exchange data simultaneously. We use a variation of the ring algorithm, where in a
 * single step a process exchanges the data with both neighbors at distance k (on the left
 * and the right on a logical ring topology). With this approach we need to pack the data
 * for a single of the two neighbors, as we can then use the original buffer (and datatype
 * and count) to send the data to the other.
 */
int
ompi_coll_base_alltoallw_intra_basic_inplace(const void *rbuf, const int *rcounts, const int *rdisps,
                                             struct ompi_datatype_t * const *rdtype,
                                             struct ompi_communicator_t *comm,
                                             const mca_coll_base_module_t *module)
{
    int i; 
    int size;
    int rank;
    int left;
    int right;
    int err = MPI_SUCCESS;
    int line;
    ompi_request_t *req = MPI_REQUEST_NULL;
    char *tmp_buffer;
    size_t packed_size = 0;
    size_t max_size;
    size_t type_size;
    opal_convertor_t convertor;

    /* Initialize. */

    size = ompi_comm_size(comm);
    if (1 == size) {
        return OMPI_SUCCESS;
    }
    rank = ompi_comm_rank(comm);

    for (i = 0, max_size = 0 ; i < size ; ++i) {
        if (i == rank) {
            continue;
        }
        ompi_datatype_type_size(rdtype[i], &type_size);
        packed_size = rcounts[i] * type_size;
        max_size = packed_size > max_size ? packed_size : max_size;
    }

    /* Easy way out */
    if (0 == max_size) {
        return MPI_SUCCESS;
    }

    /* Find the largest amount of packed send/recv data among all peers where
     * we need to pack before the send.
     */
#if OPAL_ENABLE_HETEROGENEOUS_SUPPORT
    for (i = 1 ; i <= (size >> 1) ; ++i) {
        right = (rank + i) % size;
        ompi_proc_t *ompi_proc = ompi_comm_peer_lookup(comm, right);

        if( OPAL_UNLIKELY(opal_local_arch != ompi_proc->super.proc_convertor->master->remote_arch))  {
            packed_size = opal_datatype_compute_remote_size(&rdtype[right]->super,
                                                            ompi_proc->super.proc_convertor->master->remote_sizes);
            packed_size *= rcounts[right];
            max_size = packed_size > max_size ? packed_size : max_size;
        }
    }
#endif  /* OPAL_ENABLE_HETEROGENEOUS_SUPPORT */

    /* Allocate a temporary buffer */
    tmp_buffer = calloc (max_size, 1);
    if( NULL == tmp_buffer) { err = OMPI_ERR_OUT_OF_RESOURCE; line = __LINE__; goto error_hndl; }

    for (i = 1 ; i <= (size >> 1) ; ++i) {
        struct iovec iov = {.iov_base = tmp_buffer, .iov_len = max_size};
        uint32_t iov_count = 1;

        right = (rank + i) % size;
        left  = (rank + size - i) % size;

        if( 0 != rcounts[right] ) {  /* nothing to exchange with the peer on the right */
            const ompi_proc_t *right_proc = ompi_comm_peer_lookup(comm, right);
            opal_convertor_clone(right_proc->super.proc_convertor, &convertor, 0);
            opal_convertor_prepare_for_send(&convertor, &rdtype[right]->super, rcounts[right],
                                            (const char *) rbuf + rdisps[right]);
            err = opal_convertor_pack(&convertor, &iov, &iov_count, &packed_size);
            if (1 != err) { goto error_hndl; }

            /* Receive data from the right */
            err = MCA_PML_CALL(irecv ((char *) rbuf + rdisps[right], rcounts[right], rdtype[right],
                                      right, MCA_COLL_BASE_TAG_ALLTOALLW, comm, &req));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        if( (left != right) && (0 != rcounts[left]) ) {
            /* Send data to the left */
            err = MCA_PML_CALL(send ((char *) rbuf + rdisps[left], rcounts[left], rdtype[left],
                                     left, MCA_COLL_BASE_TAG_ALLTOALLW, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) { goto error_hndl; }

            err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
            if (MPI_SUCCESS != err) { goto error_hndl; }

            /* Receive data from the left */
            err = MCA_PML_CALL(irecv ((char *) rbuf + rdisps[left], rcounts[left], rdtype[left],
                                      left, MCA_COLL_BASE_TAG_ALLTOALLW, comm, &req));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        if( 0 != rcounts[right] ) {  /* nothing to exchange with the peer on the right */
            /* Send data to the right */
            err = MCA_PML_CALL(send ((char *) tmp_buffer,  packed_size, MPI_PACKED,
                                     right, MCA_COLL_BASE_TAG_ALLTOALLW, MCA_PML_BASE_SEND_STANDARD,
                                     comm));
            if (MPI_SUCCESS != err) { goto error_hndl; }
        }

        err = ompi_request_wait (&req, MPI_STATUSES_IGNORE);
        if (MPI_SUCCESS != err) { goto error_hndl; }
    }

 error_hndl:
    /* Free the temporary buffer */
    if( NULL != tmp_buffer )
        free (tmp_buffer);

    if( MPI_SUCCESS != err ) {
        OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                     "%s:%4d\tError occurred %d, rank %2d", __FILE__, line, err,
                     rank));
        (void)line;  // silence compiler warning
    }

    /* All done */
    return err;
}

/*
 *	alltoallw_intra
 *
 *	Function:	- MPI_Alltoallw
 *	Accepts:	- same as MPI_Alltoallw()
 *	Returns:	- MPI_SUCCESS or an MPI error code
 */
int
ompi_coll_base_alltoallw_intra_basic(const void *sbuf, const int *scounts, const int *sdisps,
                                     struct ompi_datatype_t * const *sdtypes,
                                     void *rbuf, const int *rcounts, const int *rdisps,
                                     struct ompi_datatype_t * const *rdtypes,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module)
{
    int i;
    int size; 
    int rank;
    int err;
    int nreqs;
    const char *psnd;
    char *prcv;
    ompi_request_t **preq;
    ompi_request_t **reqs;
    /* Initialize. */
    if (MPI_IN_PLACE == sbuf) {
        return ompi_coll_base_alltoallw_intra_basic_inplace(rbuf, rcounts, rdisps,
                                                            rdtypes, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    /* simple optimization */

    psnd = ((const char *) sbuf) + sdisps[rank];
    prcv = ((char *) rbuf) + rdisps[rank];

    err = ompi_datatype_sndrcv(psnd, scounts[rank], sdtypes[rank],
                               prcv, rcounts[rank], rdtypes[rank]);
    if (MPI_SUCCESS != err) {
        return err;
    }

    /* If only one process, we're done. */

    if (1 == size) {
        return MPI_SUCCESS;
    }

    /* Initiate all send/recv to/from others. */

    nreqs = 0;
    reqs = preq = ompi_coll_base_comm_get_reqs(module->base_data, 2 * size);
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* Post all receives first -- a simple optimization */

    for (i = 0; i < size; ++i) {
        size_t msg_size;
        ompi_datatype_type_size(rdtypes[i], &msg_size);
        msg_size *= rcounts[i];

        if (i == rank || 0 == msg_size)
            continue;

        prcv = ((char *) rbuf) + rdisps[i];
        err = MCA_PML_CALL(irecv_init(prcv, rcounts[i], rdtypes[i],
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW, comm,
                                      preq++));
        ++nreqs;
        if (MPI_SUCCESS != err) {
            ompi_coll_base_free_reqs(reqs, nreqs);
            return err;
        }
    }

    /* Now post all sends */

    for (i = 0; i < size; ++i) {
        size_t msg_size;
        ompi_datatype_type_size(sdtypes[i], &msg_size);
        msg_size *= scounts[i];

        if (i == rank || 0 == msg_size)
            continue;

        psnd = ((const char *) sbuf) + sdisps[i];
        err = MCA_PML_CALL(isend_init(psnd, scounts[i], sdtypes[i],
                                      i, MCA_COLL_BASE_TAG_ALLTOALLW,
                                      MCA_PML_BASE_SEND_STANDARD, comm,
                                      preq++));
        ++nreqs;
        if (MPI_SUCCESS != err) {
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

int
ompi_coll_base_alltoallw_intra_pairwise(const void *sbuf, const int *scounts, const int *sdisps,
                                         struct ompi_datatype_t * const *sdtype,
                                         void* rbuf, const int *rcounts, const int *rdisps,
                                         struct ompi_datatype_t * const *rdtype,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)//NOSONAR
{
    int line = -1;
    int err = 0;
    int rank;
    int size;
    int step = 0;
    int sendto;
    int recvfrom;
    void *psnd;
    void *prcv;

    if (MPI_IN_PLACE == sbuf) {
        return ompi_coll_base_alltoallw_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                             rdtype, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:alltoallw_intra_pairwise rank %d", rank));

   /* Perform pairwise exchange starting from 1 since local exhange is done */
    for (step = 0; step < size; step++) {

        /* Determine sender and receiver for this step. */
        sendto  = (rank + step) % size;
        recvfrom = (rank + size - step) % size;

        /* Determine sending and receiving locations */
        psnd = (char* const)sbuf + (ptrdiff_t)sdisps[sendto];//NOSONAR
        prcv = (char*)rbuf + (ptrdiff_t)rdisps[recvfrom];

        /* send and receive */
        err = ompi_coll_base_sendrecv( psnd, scounts[sendto], sdtype[sendto], sendto,
                                        MCA_COLL_BASE_TAG_ALLTOALLW,
                                        prcv, rcounts[recvfrom], rdtype[recvfrom], recvfrom,
                                        MCA_COLL_BASE_TAG_ALLTOALLW,
                                        comm, MPI_STATUS_IGNORE, rank);
        if (MPI_SUCCESS != err) { line = __LINE__; goto err_hndl;  }
    }

    return MPI_SUCCESS;

 err_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "%s:%4d\tError occurred %d, rank %2d at step %d", __FILE__, line,
                 err, rank, step));
    (void)line;  // silence compiler warning
    return err;
}

/* Algorithm similar to pairwise, but this one uses non blocking send/recv and
 * it detects if it is running in a sub comm, in this case a shift is created and
 * change the start step of the pairwise, each sub comm have different shift.
 * Nonblocking send/recv are limited to avoid performance loss.
 * This limit is controlled by a MCA parameter(OMPI_MCA_coll_tuned_alltoallw_pairwise_limit).*/
int
ompi_coll_base_alltoallw_intra_shifted_pairwise(const void *sbuf, const int *scounts, const int *sdisps,
                                                struct ompi_datatype_t * const *sdtype,
                                                void* rbuf, const int *rcounts, const int *rdisps,
                                                struct ompi_datatype_t * const *rdtype,
                                                struct ompi_communicator_t *comm,
                                                mca_coll_base_module_t *module,//NOSONAR
                                                int limit)
{
    int rank;
    int size;
    int step;
    int err;
    int line;
    int wsize;
    int start = 0;
    int index_s;
    int index_r;
    const void *psnd;
    void *prcv;
    struct ompi_request_t **send_rq;
    struct ompi_request_t **recv_rq;

    if (MPI_IN_PLACE == sbuf) {
        return ompi_coll_base_alltoallw_intra_basic_inplace (rbuf, rcounts, rdisps,
                                                             rdtype, comm, module);
    }

    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    /* Check if this collective is in a sub_comm */
    wsize = ompi_comm_size(MPI_COMM_WORLD); 
    if(size < wsize) {
        int tab_ranks=0;
        int tab_wranks;
        int nb_comm;
        int shift;
        ompi_communicator_t *world = MPI_COMM_WORLD;
        /* Search the first rank in each sub comm*/
        ompi_group_translate_ranks(comm->c_local_group, 1, &tab_ranks, world->c_local_group, &tab_wranks);
        nb_comm = wsize/size;
        shift = size/nb_comm;
        /* if shift is 0 we will necessarily have the same start */
        if (shift == 0) {
            shift = 1;
        }
        /* Check distribution and create different shift for each sub comm */
        if(tab_wranks % size != 0) {
            start = shift*(tab_wranks%nb_comm) % size;
        } else {
            start = shift*(tab_wranks/size) % size;
        }
    }
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "coll:base:alltoallw_intra_shifted_pairwise rank %d", rank));

    if (limit == 0 || limit > size){
        limit = size;
    }
    recv_rq = malloc(limit * sizeof(struct ompi_request_t *));
    send_rq = malloc(limit * sizeof(struct ompi_request_t *));
    for (step = 0; step < size; step++) {
        int sendto;
        int recvfrom;
        /* Determine sender and receiver for this step. */
        sendto  = (rank + (step + start)) % size;
        recvfrom = (rank + size * 2 - (step + start)) % size;
        /* Determine sending and receiving locations */
        psnd = (const char*)sbuf + (ptrdiff_t)sdisps[sendto];
        prcv = (char*)rbuf + (ptrdiff_t)rdisps[recvfrom];

        /* Isend and Irecv until the limit is reached then wait until at least one of the messages is finished */
        if(step < limit) {
            err = MCA_PML_CALL(irecv(prcv, rcounts[recvfrom], rdtype[recvfrom],
                                     recvfrom, MCA_COLL_BASE_TAG_ALLTOALLW, comm,
                                     &recv_rq[step]));
            if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
            err = MCA_PML_CALL(isend(psnd, scounts[sendto], sdtype[sendto],
                                     sendto, MCA_COLL_BASE_TAG_ALLTOALLW,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     &send_rq[step]));
        } else {
            MPI_Waitany(limit, recv_rq, &index_r, MPI_STATUS_IGNORE);
            err = MCA_PML_CALL(irecv(prcv, rcounts[recvfrom], rdtype[recvfrom],
                                     recvfrom, MCA_COLL_BASE_TAG_ALLTOALLW, comm,
                                     recv_rq + index_r));
            if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
            MPI_Waitany(limit, send_rq, &index_s, MPI_STATUS_IGNORE);
            err = MCA_PML_CALL(isend(psnd, scounts[sendto], sdtype[sendto],
                                     sendto, MCA_COLL_BASE_TAG_ALLTOALLW,
                                     MCA_PML_BASE_SEND_STANDARD, comm,
                                     send_rq + index_s));
        }
        if (MPI_SUCCESS != err) {line = __LINE__; goto err_hndl;  }
    }
    /* Wait for completion */
    ompi_request_wait_all(limit, send_rq, MPI_STATUSES_IGNORE);
    ompi_request_wait_all(limit, recv_rq, MPI_STATUSES_IGNORE);
    free(send_rq);
    free(recv_rq);
    return MPI_SUCCESS;

 err_hndl:
    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,
                 "%s:%4d\tError occurred %d, rank %2d at step %d", __FILE__, line,
                 err, rank, step));
    (void)line;  // silence compiler warning
    return err;
}
