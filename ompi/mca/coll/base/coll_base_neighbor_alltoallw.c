/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2019 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2013      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2016 Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2017      IBM Corporation.  All rights reserved.
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
#include "ompi/mca/topo/base/base.h"

static int
ompi_coll_base_neighbor_alltoallw_cart(const void *sbuf, const int scounts[], const MPI_Aint sdisps[], // NOSONAR code complexity
                                       struct ompi_datatype_t * const *sdtypes, void *rbuf, const int rcounts[],
                                       const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtypes,
                                       struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    const mca_topo_base_comm_cart_2_2_0_t *cart = comm->c_topo->mtc.cart;
    const int rank = ompi_comm_rank (comm);
    int rc = MPI_SUCCESS;
    int dim;
    int i;
    int nreqs;
    ompi_request_t **reqs;
    ompi_request_t **preqs;

    if (0 == cart->ndims) return OMPI_SUCCESS;

    reqs = preqs = ompi_coll_base_comm_get_reqs( module->base_data, 4 * cart->ndims );
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* post receives first */
    for (dim = 0, i = 0, nreqs = 0; dim < cart->ndims ; ++dim, i += 2) {
        int srank = MPI_PROC_NULL;
        int drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }

        if (MPI_PROC_NULL != srank) {
            nreqs++;
            rc = MCA_PML_CALL(irecv((char *) rbuf + rdisps[i], rcounts[i], rdtypes[i], srank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim, comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        if (MPI_PROC_NULL != drank) {
            nreqs++;
            rc = MCA_PML_CALL(irecv((char *) rbuf + rdisps[i+1], rcounts[i+1], rdtypes[i+1], drank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1, comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs );
        return rc;
    }

    for (dim = 0, i = 0 ; dim < cart->ndims ; ++dim, i += 2) {
        int srank = MPI_PROC_NULL;
        int drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }

        if (MPI_PROC_NULL != srank) {
            nreqs++;
            /* remove cast from const when the pml layer is updated to take a const for the send buffer */
            rc = MCA_PML_CALL(isend((char *) sbuf + sdisps[i], scounts[i], sdtypes[i], srank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1, MCA_PML_BASE_SEND_STANDARD, comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }

        if (MPI_PROC_NULL != drank) {
            nreqs++;
            rc = MCA_PML_CALL(isend((char *) sbuf + sdisps[i+1], scounts[i+1], sdtypes[i+1], drank,
                                    MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim, MCA_PML_BASE_SEND_STANDARD, comm, preqs++));
            if (OMPI_SUCCESS != rc) break;
        }
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs );
        return rc;
    }

    rc = ompi_request_wait_all (nreqs, reqs, MPI_STATUSES_IGNORE);
    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, nreqs );
    }
    return rc;
}

static int
ompi_coll_base_neighbor_alltoallw_graph(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                        struct ompi_datatype_t * const sdtypes[], void *rbuf, const int rcounts[],
                                        const MPI_Aint rdisps[], struct ompi_datatype_t * const rdtypes[],
                                        struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    const mca_topo_base_comm_graph_2_2_0_t *graph = comm->c_topo->mtc.graph;
    int rc = MPI_SUCCESS;
    int neighbor;
    int degree;
    const int rank = ompi_comm_rank (comm);
    ompi_request_t **reqs;
    ompi_request_t **preqs;
    const int *edges;

    mca_topo_base_graph_neighbors_count (comm, rank, &degree);
    if (0 == degree) return OMPI_SUCCESS;

    reqs = preqs = ompi_coll_base_comm_get_reqs( module->base_data, 2 * degree );
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    edges = graph->edges;
    if (rank > 0) {
        edges += graph->index[rank - 1];
    }

    /* post all receives first */
    for (neighbor = 0; neighbor < degree ; ++neighbor) {
        rc = MCA_PML_CALL(irecv((char *) rbuf + rdisps[neighbor], rcounts[neighbor], rdtypes[neighbor],
                                edges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, comm, preqs++));
        if (OMPI_SUCCESS != rc) break;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs(reqs, neighbor + 1);
        return rc;
    }

    for (neighbor = 0 ; neighbor < degree ; ++neighbor) {
        /* remove cast from const when the pml layer is updated to take a const for the send buffer */
        rc = MCA_PML_CALL(isend((char *) sbuf + sdisps[neighbor], scounts[neighbor], sdtypes[neighbor],
                                edges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, MCA_PML_BASE_SEND_STANDARD,
                                comm, preqs++));
        if (OMPI_SUCCESS != rc) break;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs(reqs, neighbor + degree + 1);
        return rc;
    }

    rc = ompi_request_wait_all (degree * 2, reqs, MPI_STATUSES_IGNORE);
    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs(reqs, degree * 2);
    }
    return rc;
}

static int
ompi_coll_base_neighbor_alltoallw_dist_graph(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                             struct ompi_datatype_t * const *sdtypes, void *rbuf, const int rcounts[],
                                             const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtypes,
                                             struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    const mca_topo_base_comm_dist_graph_2_2_0_t *dist_graph = comm->c_topo->mtc.dist_graph;
    int rc = MPI_SUCCESS;
    int neighbor;
    const int *inedges;
    const int *outedges;
    int indegree;
    int outdegree;
    ompi_request_t **reqs;
    ompi_request_t **preqs;

    indegree = dist_graph->indegree;
    outdegree = dist_graph->outdegree;
    if( 0 == (indegree + outdegree) ) return OMPI_SUCCESS;

    inedges = dist_graph->in;
    outedges = dist_graph->out;

    if (0 == indegree+outdegree) return OMPI_SUCCESS;

    reqs = preqs = ompi_coll_base_comm_get_reqs( module->base_data, indegree + outdegree );
    if( NULL == reqs ) { return OMPI_ERR_OUT_OF_RESOURCE; }

    /* post all receives first */
    for (neighbor = 0; neighbor < indegree ; ++neighbor) {
        rc = MCA_PML_CALL(irecv((char *) rbuf + rdisps[neighbor], rcounts[neighbor], rdtypes[neighbor],
                                inedges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, comm, preqs++));
        if (OMPI_SUCCESS != rc) break;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs(reqs, neighbor + 1);
        return rc;
    }

    for (neighbor = 0 ; neighbor < outdegree ; ++neighbor) {
        /* remove cast from const when the pml layer is updated to take a const for the send buffer */
        rc = MCA_PML_CALL(isend((char *) sbuf + sdisps[neighbor], scounts[neighbor], sdtypes[neighbor],
                                outedges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, MCA_PML_BASE_SEND_STANDARD,
                                comm, preqs++));
        if (OMPI_SUCCESS != rc) break;
    }

    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs(reqs, indegree + neighbor + 1);
        return rc;
    }

    rc = ompi_request_wait_all (indegree + outdegree, reqs, MPI_STATUSES_IGNORE);
    if (OMPI_SUCCESS != rc) {
        ompi_coll_base_free_reqs( reqs, indegree + outdegree );
    }
    return rc;
}

static int
ompi_coll_base_controlled_neighbor_alltoallw_dist_graph(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                                        struct ompi_datatype_t * const *sdtype, void *rbuf, const int rcounts[],
                                                        const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtype,
                                                        struct ompi_communicator_t *comm, const mca_coll_base_module_t *module,
                                                        int limit)//NOSONAR
{
    const mca_topo_base_comm_dist_graph_2_2_0_t *dist_graph = comm->c_topo->mtc.dist_graph;
    int err;
    int neighbor;
    const int *inedges;
    const int *outedges;
    int indegree;
    int outdegree;
    int send_index;
    ompi_request_t **send_rq;
    ompi_request_t **recv_rq;

    /* Check number of send and recv */
    indegree = dist_graph->indegree;
    outdegree = dist_graph->outdegree;
    if(0 == (indegree + outdegree)) {
        return OMPI_SUCCESS;
    }

    /* Retrieve sender and receiver */
    inedges = dist_graph->in;
    outedges = dist_graph->out;
    if (outdegree < limit) {
        limit  = outdegree;
    }

    recv_rq = malloc(indegree * sizeof(struct ompi_request_t));
    send_rq = malloc(limit * sizeof(struct ompi_request_t));

    /* post all receives first */
    for (neighbor = 0; neighbor < indegree ; ++neighbor) {
        err = MCA_PML_CALL(irecv((char *) rbuf + rdisps[neighbor], rcounts[neighbor], rdtype[neighbor],
                           inedges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, comm, &recv_rq[neighbor]));
        if (OMPI_SUCCESS != err) goto end_err;
    }

    for (neighbor = 0 ; neighbor < outdegree ; ++neighbor) {
        if (neighbor < limit) {
            err = MCA_PML_CALL(isend((char *) sbuf + sdisps[neighbor], scounts[neighbor], sdtype[neighbor],
                               outedges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, MCA_PML_BASE_SEND_STANDARD,
                               comm, &send_rq[neighbor]));
        } else {
            MPI_Waitany(limit, send_rq, &send_index, MPI_STATUS_IGNORE);
            err = MCA_PML_CALL(isend((char *) sbuf + sdisps[neighbor], scounts[neighbor], sdtype[neighbor],
                                outedges[neighbor], MCA_COLL_BASE_TAG_NEIGHBOR_BASE, MCA_PML_BASE_SEND_STANDARD,
                                comm, send_rq + send_index));
        }
        if (OMPI_SUCCESS != err) goto end_err;
    }
    ompi_request_wait_all(limit, send_rq, MPI_STATUSES_IGNORE);
    ompi_request_wait_all(indegree, recv_rq, MPI_STATUSES_IGNORE);
    free(send_rq);
    free(recv_rq);
    return OMPI_SUCCESS;
end_err:
    free(send_rq);
    free(recv_rq);
    return err;
}

static int
ompi_coll_base_controlled_neighbor_alltoallw_cart(const void *sbuf, const int scounts[], const MPI_Aint sdisps[], // NOSONAR code complexity
                                                  struct ompi_datatype_t * const *sdtype, void *rbuf, const int rcounts[],
                                                  const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtype,
                                                  struct ompi_communicator_t *comm, const mca_coll_base_module_t *module,
                                                  int limit)//NOSONAR
{
    const mca_topo_base_comm_cart_2_2_0_t *cart = comm->c_topo->mtc.cart;
    const int rank = ompi_comm_rank (comm);
    int err;
    int dim;
    int peer;
    int nb_rrq = 0;
    int nb_srq = 0;
    int send_index;
    ompi_request_t **send_rq;
    ompi_request_t **recv_rq;
    if( 0 == cart->ndims ) return OMPI_SUCCESS;

    recv_rq = malloc(2 * cart->ndims * sizeof(struct ompi_request_t));
    send_rq = malloc(limit * sizeof(struct ompi_request_t));
    /* post receives first */
    for (dim = 0, peer = 0; dim < cart->ndims ; ++dim, peer += 2) {
        int srank = MPI_PROC_NULL;
        int drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }
        if (MPI_PROC_NULL != srank) {
            err = MCA_PML_CALL(irecv((char *) rbuf + rdisps[peer], rcounts[peer], rdtype[peer], srank,
                                     MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim, comm, &recv_rq[nb_rrq]));
            nb_rrq++;
            if (OMPI_SUCCESS != err) goto end_err;
        }
        if (MPI_PROC_NULL != drank) {
            err = MCA_PML_CALL(irecv((char *) rbuf + rdisps[peer+1], rcounts[peer+1], rdtype[peer], drank,
                                     MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1, comm, &recv_rq[nb_rrq]));
            nb_rrq++;
            if (OMPI_SUCCESS != err) goto end_err;
        }
    }

    for (dim = 0, peer = 0 ; dim < cart->ndims ; ++dim, peer += 2) {
        int srank = MPI_PROC_NULL;
        int drank = MPI_PROC_NULL;

        if (cart->dims[dim] > 1) {
            mca_topo_base_cart_shift (comm, dim, 1, &srank, &drank);
        } else if (1 == cart->dims[dim] && cart->periods[dim]) {
            srank = drank = rank;
        }

        if (MPI_PROC_NULL != srank) {
            if (nb_srq < limit) {
                err = MCA_PML_CALL(isend((char *) sbuf + sdisps[peer], scounts[peer], sdtype[peer], srank,
                                   MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1, MCA_PML_BASE_SEND_STANDARD, comm, &send_rq[nb_srq]));
                nb_srq++;
                if (OMPI_SUCCESS != err) goto end_err;
            } else {
                MPI_Waitany(limit, send_rq, &send_index, MPI_STATUS_IGNORE);
                err = MCA_PML_CALL(isend((char *) sbuf + sdisps[peer], scounts[peer], sdtype[peer], srank,
                                   MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim - 1, MCA_PML_BASE_SEND_STANDARD, comm, send_rq + send_index));
                if (OMPI_SUCCESS != err) goto end_err;
            }
        }

        if (MPI_PROC_NULL != drank) {
            if (nb_srq < limit) {
                err = MCA_PML_CALL(isend((char *) sbuf + sdisps[peer+1], scounts[peer+1], sdtype[peer+1], drank,
                                         MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim, MCA_PML_BASE_SEND_STANDARD, comm, &send_rq[nb_srq]));
                nb_srq++;
                if (OMPI_SUCCESS != err) goto end_err;
            } else {
                MPI_Waitany(limit, send_rq, &send_index, MPI_STATUS_IGNORE);
                err = MCA_PML_CALL(isend((char *) sbuf + sdisps[peer+1], scounts[peer+1], sdtype[peer+1], drank,
                                         MCA_COLL_BASE_TAG_NEIGHBOR_BASE - 2 * dim, MCA_PML_BASE_SEND_STANDARD, comm, send_rq + send_index));
                if (OMPI_SUCCESS != err) goto end_err;
            }
        }
    }
    ompi_request_wait_all(nb_srq, send_rq, MPI_STATUSES_IGNORE);
    ompi_request_wait_all(nb_rrq, recv_rq, MPI_STATUSES_IGNORE);
    free(send_rq);
    free(recv_rq);
    return OMPI_SUCCESS;
end_err:
    free(send_rq);
    free(recv_rq);
    return err;
}

int ompi_coll_base_neighbor_alltoallw(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                      struct ompi_datatype_t * const *sdtypes, void *rbuf, const int rcounts[],
                                      const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtypes,
                                      struct ompi_communicator_t *comm, mca_coll_base_module_t *module)
{
    if (OMPI_COMM_IS_INTER(comm)) {
        return OMPI_ERR_NOT_SUPPORTED;
    }

    if (OMPI_COMM_IS_CART(comm)) {
        return ompi_coll_base_neighbor_alltoallw_cart (sbuf, scounts, sdisps, sdtypes, rbuf,
                                                       rcounts, rdisps, rdtypes, comm, module);
    } else if (OMPI_COMM_IS_GRAPH(comm)) {
        return ompi_coll_base_neighbor_alltoallw_graph (sbuf, scounts, sdisps, sdtypes, rbuf,
                                                        rcounts, rdisps, rdtypes, comm, module);
    } else if (OMPI_COMM_IS_DIST_GRAPH(comm)) {
        return ompi_coll_base_neighbor_alltoallw_dist_graph (sbuf, scounts, sdisps, sdtypes, rbuf,
                                                             rcounts, rdisps, rdtypes, comm, module);
    }

    return OMPI_ERR_NOT_SUPPORTED;
}
int ompi_coll_base_neighbor_alltoallw_controlled(const void *sbuf, const int scounts[], const MPI_Aint sdisps[],
                                                 struct ompi_datatype_t * const *sdtype, void *rbuf, const int rcounts[],
                                                 const MPI_Aint rdisps[], struct ompi_datatype_t * const *rdtype,
                                                 struct ompi_communicator_t *comm, mca_coll_base_module_t *module,
                                                 int limit)
{
    if (OMPI_COMM_IS_INTER(comm)) {
        return OMPI_ERR_NOT_SUPPORTED;
    }
    if (OMPI_COMM_IS_DIST_GRAPH(comm)) {
        return ompi_coll_base_controlled_neighbor_alltoallw_dist_graph(sbuf, scounts, sdisps, sdtype, rbuf,
                                                                       rcounts, rdisps, rdtype, comm, module,
                                                                       limit);
    } else if (OMPI_COMM_IS_CART(comm)) {
        return ompi_coll_base_controlled_neighbor_alltoallw_cart(sbuf, scounts, sdisps, sdtype, rbuf,
                                                                 rcounts, rdisps, rdtype, comm, module,
                                                                 limit);
    }
    return ompi_coll_base_neighbor_alltoallw(sbuf, scounts, sdisps, sdtype,
                                             rbuf, rcounts, rdisps, rdtype,
                                             comm, module);
}
