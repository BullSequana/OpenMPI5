/*
 * Copyright (c) 2018-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "coll_han.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_tags.h"
#include "ompi/mca/pml/pml.h"
#include "coll_han_trigger.h"

static int mca_coll_han_ml_alltoall_grid_map_by_core(const void *sbuf, int scount,
                                                     struct ompi_datatype_t *sdtype,
                                                     void *rbuf, int rcount,
                                                     struct ompi_datatype_t *rdtype,
                                                     struct ompi_communicator_t *comm,
                                                     mca_coll_han_module_t *han_module);

/* mca_coll_han_alltoall_intra_grid
 * Perform an alltoall operation using a intra-node alltoallw
 * followed by an inter-node alltoallw (or alltoall in block distribution)
 * Avoid reordering through datatypes
 *
 * Ex: distribution of 6 ranks over 3 nodes
 * global_rank(up_rank,low_rank)
 * 0(0,0)    1(1,0)    3(2,0)
 * 2(0,1)    4(1,1)    5(2,1)
 *
 * Buffer representation: source destination
 * Buffer 32 means message from 3 to 2
 *
 * Initial state:
 * 0: 00 01 02 03 04 05
 * 1: 10 11 12 13 14 15
 * 2: 20 21 22 23 24 25
 * 3: 30 31 32 33 34 35
 * 4: 40 41 42 43 44 45
 * 5: 50 51 52 53 54 55
 *
 * First alltoallw: send every messages destined to a MPI task
 * with low_rank = low to the MPI task in our node with low_rank = low
 * Do not pack them, use datatypes instead
 *
 * X = sent/recv buffer
 * _ = not sent message
 * Datatypes application on send buffer:
 * low = 0: XX_X__
 * low = 1: __X_XX
 *
 * Use vector on receive side to pack buffers
 * Use byte displacement of alltoallw to intertwine vectors
 * Data from low = 0: X_X_X_
 * Data from low = 1: _X_X_X
 *
 * Intemediary buffer after first alltoallw:
 * 0: 00 20 01 21 03 23
 * 2: 02 22 04 24 05 25
 *
 * 1: 10 40 11 41 13 43
 * 4: 12 42 14 44 15 45
 *
 * 3: 30 50 31 51 33 53
 * 5: 32 52 34 54 35 55
 *
 * Second alltoallw: send messages to their destination
 * Data are already packed, send data as bytes
 * Use complex datatypes on receiving side to avoid reordering
 * Cannot use vector here: stride can vary between blocks
 * Datatypes application on recv buffer
 * up = 0: X_X___
 * up = 1: _X__X_
 * up = 2: ___X_X
 * Note that block here are low_size times bigger than for the first alltoallw
 *
 * Receive buffer after second alltoallw:
 * 0: 00 10 20 30 40 50
 * 1: 01 11 21 31 41 51
 * 2: 02 12 22 32 42 52
 * 3: 03 13 23 33 43 53
 * 4: 04 14 24 34 44 54
 * 5: 05 15 25 35 45 55
 *
 * Limitations:
 *     - Only for balanced ppn over nodes
 */
int mca_coll_han_alltoall_intra_grid(const void *sbuf, int scount,
                                     struct ompi_datatype_t *sdtype,
                                     void *rbuf, int rcount,
                                     struct ompi_datatype_t *rdtype,
                                     struct ompi_communicator_t *comm,
                                     mca_coll_base_module_t *module)
{
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoall);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    int ret = mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    if( OMPI_SUCCESS != ret ) {  /* Let's hope the error is consistently returned across the entire communicator */
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    } 

    /* setup up/low and global coordinates */
    int w_size = ompi_comm_size(comm);

    /* Init topo */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoall with this communicator. It need to fall back on another component\n"));
        return han_module->previous_alltoall(sbuf, scount, sdtype,
                                             rbuf, rcount, rdtype,
                                             comm, han_module->previous_alltoall_module);
    }

    const char *real_sbuf;

    /* MPI_IN_PLACE cases */
    if (MPI_IN_PLACE == sbuf) {
        scount = rcount;
        sdtype = rdtype;
        real_sbuf = rbuf;
    } else {
        real_sbuf = sbuf;
    }

    int *global_ranks = han_module->global_ranks;
    int max_low_size = han_module->maximum_size[LEAF_LEVEL];

    /* Loop iterators */
    int up, low;

    /* setup up/low and global coordinates */
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);
    int max_size;

    if (low_size > up_size) {
        max_size = low_size;
    } else {
        max_size = up_size;
    }

    /* allocate the intermediary parameter buffers and datatypes
     * to perform the low and up alltoall collective */
    int *low_sendcounts, *low_recvcounts, *low_sdispls, *low_rdispls;
    int *up_sendcounts, *up_recvcounts, *up_sdispls, *up_rdispls;

    low_sendcounts = (int *)malloc(low_size * sizeof(int));
    low_recvcounts = (int *)malloc(low_size * sizeof(int));
    low_sdispls = (int *)malloc(low_size * sizeof(int));
    low_rdispls = (int *)malloc(low_size * sizeof(int));

    if (NULL == low_sendcounts
        || NULL == low_recvcounts
        || NULL == low_sdispls
        || NULL == low_rdispls) {
        return OMPI_ERROR;
    }

    struct ompi_datatype_t **low_sendtypes, **low_recvtypes;
    struct ompi_datatype_t **up_sendtypes, **up_recvtypes;
    low_sendtypes = (struct ompi_datatype_t **)malloc(low_size * sizeof(struct ompi_datatype_t *));
    low_recvtypes = (struct ompi_datatype_t **)malloc(low_size * sizeof(struct ompi_datatype_t *));

    if (NULL == low_sendtypes || NULL == low_recvtypes) {
        free(low_rdispls);
        free(low_sdispls);
        free(low_recvcounts);
        free(low_sendcounts);
        return OMPI_ERROR;
    }

    char *up_ata_buf = NULL;
    char *up_ata_buf_start = NULL;
    char *low_ata_buf = NULL;
    char *low_ata_buf_start = NULL;

    ptrdiff_t sextent;
    size_t sdtype_size;
    ompi_datatype_type_extent(sdtype, &sextent);
    ompi_datatype_type_size(sdtype, &sdtype_size);

    /* intermediary buffer on node leaders to gather on low comm */
    low_ata_buf = (char *)malloc(sdtype_size * scount * w_size);

    /* We use a vector datatype to pack the intermediary buffer per destination */
    struct ompi_datatype_t *vector_sdtype, *vector_rdtype;
    ompi_datatype_create_vector(up_size, scount * sdtype_size, low_size * scount * sdtype_size, MPI_BYTE, &vector_rdtype);
    ompi_datatype_commit(&vector_rdtype);

    if (!han_module->is_mapbycore) {

        int *displacement;
        displacement = (int *)malloc(max_size * sizeof(int));

        for (low = 0 ; low < low_size ; low++) {
            for (up = 0 ; up < up_size ; up++) {
                displacement[up] = scount * global_ranks[up * max_low_size + low];
            }

            ompi_datatype_create_indexed_block(up_size, scount, displacement, sdtype, &(low_sendtypes[low]));
            ompi_datatype_commit(&(low_sendtypes[low]));

            low_sendcounts[low] = 1;
            low_recvcounts[low] = 1;
            low_sdispls[low] = 0;
            low_rdispls[low] = low * scount * sdtype_size;
            low_recvtypes[low] = vector_rdtype;
        }

        up_sendcounts = malloc(up_size * sizeof(int));
        up_recvcounts = malloc(up_size * sizeof(int));
        up_sdispls = malloc(up_size * sizeof(int));
        up_rdispls = malloc(up_size * sizeof(int));
        up_sendtypes = malloc(up_size * sizeof(struct ompi_datatype_t *));
        up_recvtypes = malloc(up_size * sizeof(struct ompi_datatype_t *));

        for (up = 0 ; up < up_size ; up++) {
            for (low = 0 ; low < low_size ; low++) {
                displacement[low] = rcount * global_ranks[up * max_low_size + low];
            }

            ompi_datatype_create_indexed_block(low_size, rcount, displacement, rdtype, &(up_recvtypes[up]));
            ompi_datatype_commit(&(up_recvtypes[up]));

            up_sendcounts[up] = scount * sdtype_size * low_size;
            up_sdispls[up] = up_sendcounts[up] * up;
            up_sendtypes[up] = MPI_BYTE;

            up_recvcounts[up] = 1;
            up_rdispls[up] = 0;
        }

        free(displacement);

    } else {
        ompi_datatype_create_vector(up_size, scount, low_size * scount, sdtype, &vector_sdtype);
        ompi_datatype_commit(&vector_sdtype);
        for (low = 0 ; low < low_size ; low++) {
            low_sendcounts[low] = 1;
            low_recvcounts[low] = 1;
            low_sdispls[low] = low * scount * sextent;
            low_rdispls[low] = low * scount * sdtype_size;
            low_sendtypes[low] = vector_sdtype;
            low_recvtypes[low] = vector_rdtype;
        }
    }

    /* Low comm alltoallw
     * the use of alltoallw allows us to interwind the vectors 
     * directly and avoid a further reordering to organize the data 
     */
    low_comm->c_coll->coll_alltoallw((char *)real_sbuf, low_sendcounts, low_sdispls, low_sendtypes,
                                     low_ata_buf,
                                     low_recvcounts, low_rdispls, low_recvtypes,
                                     low_comm,
                                     low_comm->c_coll->coll_alltoallw_module);

    /* Up comm alltoall
     * Use alltoallw if not map by core to avoid reordering
     */
    if (han_module->is_mapbycore) {
        up_comm->c_coll->coll_alltoall(low_ata_buf, low_size * scount * sdtype_size,
                                       MPI_BYTE, rbuf,
                                       low_size * rcount, rdtype,
                                       up_comm,
                                       up_comm->c_coll->coll_alltoall_module);
    } else {
        up_comm->c_coll->coll_alltoallw(low_ata_buf,
                                        up_sendcounts, up_sdispls, up_sendtypes,
                                        rbuf,
                                        up_recvcounts, up_rdispls, up_recvtypes,
                                        up_comm,
                                        up_comm->c_coll->coll_alltoallw_module);
    }

    if (!han_module->is_mapbycore) {
        for (low = 0 ; low < low_size ; low++) {
            ompi_datatype_destroy(&(low_sendtypes[low]));
        }

        for (up = 0 ; up < up_size ; up++) {
            ompi_datatype_destroy(&(up_recvtypes[up]));
        }

        free(up_recvtypes);
        free(up_sendtypes);
        free(up_rdispls);
        free(up_sdispls);
        free(up_recvcounts);
        free(up_sendcounts);
    } else {
        ompi_datatype_destroy(&vector_sdtype);
    }
    ompi_datatype_destroy(&vector_rdtype);
    free(low_ata_buf);

    free(low_recvtypes);
    free(low_sendtypes);
    free(low_rdispls);
    free(low_sdispls);
    free(low_recvcounts);
    free(low_sendcounts);

    return OMPI_SUCCESS;
}

/* mca_coll_han_alltoall_intra_pipelined_grid
 * Pipelined version of grid algorithm
 *
 * Example of an ideal pipeline for messages segmented in 4 parts:
 *  __________________________________
 * |Step          | 0 | 1 | 2 | 3 | 4 |
 * |______________|___|___|___|___|___|
 * |low_alltoallw | 0 | 1 | 2 | 3 | X |
 * |______________|___|___|___|___|___|
 * |up_ialltoallw | X | 0 | 1 | 2 | 3 |
 * |______________|___|___|___|___|___|
 *
 * As the pipeline is not explicit, the dynamic can vary. Here are the dependencies:
 * low_alltoallw of segment i depends on up_ialltoallw of segment i-2
 * up_ialltoallw of segment i depends on low_alltoallw of segment i
 *
 * The last segment may not be full. It is handle by the rest structures
 *
 * Limitations:
 *     - Only for balanced ppn over nodes
 */
int mca_coll_han_alltoall_intra_pipelined_grid(const void *sbuf, int scount,
                                               struct ompi_datatype_t *sdtype,
                                               void *rbuf, int rcount,
                                               struct ompi_datatype_t *rdtype,
                                               struct ompi_communicator_t *comm,
                                               mca_coll_base_module_t *module)
{
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoall);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    int ret = mca_coll_han_comm_create_new(comm, han_module);
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    if( OMPI_SUCCESS != ret ) {  /* Let's hope the error is consistently returned across the entire communicator */
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    } 

    /* Init topo */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* Unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoall with this communicator. It need to fall back on another component\n"));
        return han_module->previous_alltoall(sbuf, scount, sdtype,
                                             rbuf, rcount, rdtype,
                                             comm, han_module->previous_alltoall_module);
    }

    /* MPI_IN_PLACE cases */
    const char *real_sbuf;
    char *alloc = NULL;
    int w_size = ompi_comm_size(comm);

    if (MPI_IN_PLACE == sbuf) {
        ptrdiff_t span, gap;
        span = opal_datatype_span(&rdtype->super, rcount * w_size, &gap);
        scount = rcount;
        sdtype = rdtype;
        alloc = (char*) malloc(span);
        ompi_datatype_copy_content_same_ddt(rdtype, rcount * w_size,
                                            alloc - gap, rbuf);
        real_sbuf = alloc - gap;
    } else {
        real_sbuf = sbuf;
    }

    /* Aliases */
    int max_low_size = han_module->maximum_size[LEAF_LEVEL];
    int *global_ranks = han_module->global_ranks;

    /* Communicators infos */
    int low_size, up_size;
    int dest;

    /* User messages infos */
    ptrdiff_t sdtype_extent, rdtype_extent;
    size_t sdtype_size, rdtype_size;

    /* Segmentation informations */
    int ssegsize, rsegsize, nseg, rest;

    /* Datatype creation infos */
    int *displs;

    /* Alltoallw infos */
    struct ata_infos {
        int *scounts, *sdispls;
        int *rcounts, *rdispls;
        struct ompi_datatype_t **stypes, **rtypes;
    } low_ata, up_ata;
    struct ompi_datatype_t **rest_low_stypes, **rest_up_stypes, **rest_up_rtypes;
    int *rest_scounts, *rest_sdispls;
    struct ompi_datatype_t *main_vector, *rest_vector;

    /* Intermediate buffers
     * Use 2 of them to overlap ialltoallw with alltoallw
     */
    void *tmpbuf[2];
    struct ompi_request_t *rq[2];

    /* Loop iterators */
    int seg, up, low;

    /* Get communicators infos */
    up_size = ompi_comm_size(up_comm);
    low_size = ompi_comm_size(low_comm);

    /* Get user messages infos */
    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);

    ompi_datatype_type_size(sdtype, &sdtype_size);
    ompi_datatype_type_size(rdtype, &rdtype_size);

    /* Compute segmentation informations */
    ssegsize = scount;
    rsegsize = rcount;

    /* Activate segmentation if the scount is high enough
     * Do not cut datatypes
     */
    if (scount == rcount) {
        ssegsize = mca_coll_han_component.han_alltoall_segsize/sdtype_size;

        /* Handle sdtype_size > segsize */
        if (0 == ssegsize) {
            ssegsize = 1;
        }

        /* Handle full message < segsize */
        if (ssegsize > scount) {
            ssegsize = scount;
        }

        rsegsize = ssegsize;
    }
    nseg = scount / ssegsize;
    rest = scount % ssegsize;
 
    /* Displs allocataion
     * To reuse it, allocate the maximum between low_size and up_size
     */
    if (low_size > up_size) {
        displs = malloc(low_size * sizeof(int));
    } else {
        displs = malloc(up_size * sizeof(int));
    }

    /* Compute low alltoallw infos */
    low_ata.scounts = malloc(low_size * sizeof(int));
    low_ata.sdispls = malloc(low_size * sizeof(int));
    low_ata.rcounts = malloc(low_size * sizeof(int));
    low_ata.rdispls = malloc(low_size * sizeof(int));
    low_ata.stypes = malloc(low_size * sizeof(struct ompi_datatype_t *));
    low_ata.rtypes = malloc(low_size * sizeof(struct ompi_datatype_t *));
    if (rest) {
        rest_low_stypes = malloc(low_size * sizeof(struct ompi_datatype_t *));
    }
    for (low = 0 ; low < low_size ; low++) {
        int inter_node_root;

        /* Avoid an offset at the start of the datatype */
        inter_node_root = global_ranks[0 * max_low_size + low];

        low_ata.rcounts[low] = ssegsize * sdtype_size * up_size;
        low_ata.rdispls[low] = ssegsize * sdtype_size * up_size * low;
        low_ata.rtypes[low] = MPI_BYTE;

        low_ata.scounts[low] = 1;
        low_ata.sdispls[low] = inter_node_root * scount * sdtype_extent;

        for (up = 0 ; up < up_size ; up++) {
            dest = global_ranks[up * max_low_size + low];
            displs[up] = (dest - inter_node_root) * scount;
        }

        ompi_datatype_create_indexed_block(up_size,
                                           ssegsize,
                                           displs,
                                           sdtype,
                                           &(low_ata.stypes[low]));
        ompi_datatype_commit(&(low_ata.stypes[low]));
        if (rest) {
            ompi_datatype_create_indexed_block(up_size,
                                               rest,
                                               displs,
                                               sdtype,
                                               &(rest_low_stypes[low]));
            ompi_datatype_commit(&(rest_low_stypes[low]));
        }
    }

    /* Compute up ialltoallw infos */
    ompi_datatype_create_vector(low_size,
                                rsegsize*rdtype_size,
                                rsegsize*rdtype_size*up_size,
                                MPI_BYTE,
                                &main_vector);
    ompi_datatype_commit(&main_vector);
    if (rest) {
        ompi_datatype_create_vector(low_size,
                                    rest*rdtype_size,
                                    rest*rdtype_size*up_size,
                                    MPI_BYTE,
                                    &rest_vector);
        ompi_datatype_commit(&rest_vector);
    }

    up_ata.scounts = malloc(up_size * sizeof(int));
    up_ata.sdispls = malloc(up_size * sizeof(int));
    up_ata.rcounts = malloc(up_size * sizeof(int));
    up_ata.rdispls = malloc(up_size * sizeof(int));
    up_ata.stypes = malloc(up_size * sizeof(struct ompi_datatype_t *));
    up_ata.rtypes = malloc(up_size * sizeof(struct ompi_datatype_t *));
    if (rest) {
        rest_scounts = malloc(up_size * sizeof(int));
        rest_sdispls = malloc(up_size * sizeof(int));
        rest_up_stypes = malloc(up_size * sizeof(struct ompi_datatype_t *));
        rest_up_rtypes = malloc(up_size * sizeof(struct ompi_datatype_t *));
    }
    for (up = 0 ; up < up_size ; up++) {
        int node_root;

        /* Avoid an offset at the start of the datatype */
        node_root = global_ranks[up * max_low_size + 0];

        up_ata.scounts[up] = 1;
        up_ata.sdispls[up] = up * rsegsize * rdtype_size;
        up_ata.stypes[up] = main_vector;

        up_ata.rcounts[up] = 1;
        up_ata.rdispls[up] = node_root * rcount * rdtype_extent;

        for (low = 0 ; low < low_size ; low++) {
            dest = global_ranks[up * max_low_size + low];
            displs[low] = (dest - node_root) * rcount;
        }

        ompi_datatype_create_indexed_block(low_size,
                                           rsegsize,
                                           displs,
                                           rdtype,
                                           &(up_ata.rtypes[up]));
        ompi_datatype_commit(&(up_ata.rtypes[up]));
        if (rest) {
            rest_scounts[up] = rest * sdtype_size * low_size;
            rest_sdispls[up] = rest * sdtype_size * low_size;
            rest_up_stypes[up] = rest_vector;
            ompi_datatype_create_indexed_block(low_size,
                                               rest,
                                               displs,
                                               rdtype,
                                               &(rest_up_rtypes[up]));
            ompi_datatype_commit(&(rest_up_rtypes[up]));
        }
    }

    /* Compute intermediate buffer infos */
    tmpbuf[0] = malloc(ssegsize * sdtype_size * low_size * up_size);
    tmpbuf[1] = malloc(ssegsize * sdtype_size * low_size * up_size);
    rq[0] = rq[1] = MPI_REQUEST_NULL;

    /* Perform communications */
    for (seg = 0 ; seg < nseg ; seg++) {
        /* Perform alltoallw on intra node communicator for this segment */
        low_comm->c_coll->coll_alltoallw(real_sbuf + seg*ssegsize*sdtype_extent,
                                         low_ata.scounts,
                                         low_ata.sdispls,
                                         low_ata.stypes,
                                         tmpbuf[seg%2],
                                         low_ata.rcounts,
                                         low_ata.rdispls,
                                         low_ata.rtypes,
                                         low_comm,
                                         low_comm->c_coll->coll_alltoallw_module);

        /* Initiate ialltoallw on inter node communicator for this segment */
        up_comm->c_coll->coll_ialltoallw(tmpbuf[seg%2],
                                         up_ata.scounts,
                                         up_ata.sdispls,
                                         up_ata.stypes,
                                         ((char*) rbuf) + seg*rsegsize*rdtype_extent,
                                         up_ata.rcounts,
                                         up_ata.rdispls,
                                         up_ata.rtypes,
                                         up_comm,
                                         &(rq[seg%2]),
                                         up_comm->c_coll->coll_ialltoallw_module);

        /* Wait for the previous segment ialltoallw completion */
        ompi_request_wait(&(rq[(seg+1)%2]), MPI_STATUS_IGNORE);
    }

    /* Pipeline completion */
    if (rest) {

        for (low = 0 ; low < low_size ; low++) {
            low_ata.rcounts[low] = rest * sdtype_size;
            low_ata.rdispls[low] = rest * sdtype_size;
        }

        /* Perform alltoallw on intra node communicator for the rest */
        low_comm->c_coll->coll_alltoallw(real_sbuf + seg*ssegsize*sdtype_extent,
                                         low_ata.scounts,
                                         low_ata.sdispls,
                                         rest_low_stypes,
                                         tmpbuf[seg%2],
                                         low_ata.rcounts,
                                         low_ata.rdispls,
                                         low_ata.rtypes,
                                         low_comm,
                                         low_comm->c_coll->coll_alltoallw_module);

        /* Perform alltoallw on inter node communicator for the rest
         * No need to use ialltoallw: this is the last segment
         */
        up_comm->c_coll->coll_alltoallw(tmpbuf[seg%2],
                                        rest_scounts,
                                        rest_sdispls,
                                        rest_up_stypes,
                                        ((char*) rbuf) + seg*rsegsize*rdtype_extent,
                                        up_ata.rcounts,
                                        up_ata.rdispls,
                                        rest_up_rtypes,
                                        up_comm,
                                        up_comm->c_coll->coll_ialltoallw_module);
    }

    /* Wait remaining ialltoallw completion */
    ompi_request_wait_all(2, rq, MPI_STATUSES_IGNORE);

    /* Free buffers */
    free(tmpbuf[1]);
    free(tmpbuf[0]);

    /* Destroy datatypes */
    for (low = 0 ; low < low_size ; low++) {
        ompi_datatype_destroy(&(low_ata.stypes[low]));
        if (rest) {
            ompi_datatype_destroy(&(rest_low_stypes[low]));
        }
    }
    for (up = 0 ; up < up_size ; up++) {
        ompi_datatype_destroy(&(up_ata.rtypes[up]));
        if (rest) {
            ompi_datatype_destroy(&(rest_up_rtypes[up]));
        }
    }
    free(displs);

    /* Free rest types */
    ompi_datatype_destroy(&main_vector);
    if (rest) {
        ompi_datatype_destroy(&rest_vector);

        free(rest_up_rtypes);
        free(rest_up_stypes);
        free(rest_low_stypes);
    }

    /* Free up_comm ialltoallw ressources */
    free(up_ata.rtypes);
    free(up_ata.rdispls);
    free(up_ata.rcounts);
    free(up_ata.stypes);
    free(up_ata.sdispls);
    free(up_ata.scounts);

    /* Free low_comm alltoallw ressources */
    free(low_ata.rtypes);
    free(low_ata.rdispls);
    free(low_ata.rcounts);
    free(low_ata.stypes);
    free(low_ata.sdispls);
    free(low_ata.scounts);

    if (NULL != alloc) {
        free(alloc);
    }

    return OMPI_SUCCESS;
}

/* mca_coll_han_alltoall_intra_rolling_igatherw
 *
 * The goal here is to split the up_alltoall of the grid algorithm
 *
 * Ex: distribution of 6 ranks over 3 nodes
 * global_rank(up_rank,low_rank)
 * 0(0,0)    1(1,0)    3(2,0)
 * 2(0,1)    4(1,1)    5(2,1)
 *
 * There are 3 nodes so there will be 3 steps.
 * At each step, we identify a rank on each up communicator
 * which will receive its data. To avoid contention on a node,
 * we spread chosen ranks over nodes.
 *
 * For the first step, the chosen ranks are:
 * >0<    1     3
 *  2    >4<    5
 *
 * The first operation on this step is to gather data over
 * low_communicators (via an alltoallw) to be able to send
 * all the messages destinated to the chosen ranks
 * using up_communicators
 *
 * 0 will gather data from 0 and 2 destinated to 0
 * 2 will gather data from 0 and 2 destinated to 4
 *
 * 1 will gather data from 1 and 4 destinated to 0
 * 4 will gather data from 1 and 4 destinated to 4
 *
 * 3 will gather data from 3 and 5 destinated to 0
 * 5 will gather data from 3 and 5 destinated to 4
 *
 * These gathers are done at the same time through an alltoallw on low_communicator.
 *
 * The data is then gathered to the selected ranks using igatherw.
 * We use derivated datatypes (indexed block) to avoid data reordering.
 * We use non-blocking collectives on up_communicators to overlap the next step
 * blocking collective comunication on low_communicator.
 *
 * On the next steps, we rotate the chosen ranks over nodes.
 * The chosen ranks for step 2 are:
 *  0    >1<    3
 *  2     4    >5<
 *
 *  And for the step 3:
 *  0     1    >3<
 * >2<    4     5
 *
 * After the last step, each rank have been the root of an up_igatherw
 * exactly once, allowing it to receive its destinated data.
 *
 * Limitations:
 *     - Only for balanced ppn over nodes
 */
int mca_coll_han_alltoall_intra_rolling_igatherw(const void *sbuf, int scount,
                                                 struct ompi_datatype_t *sdtype,
                                                 void *rbuf, int rcount,
                                                 struct ompi_datatype_t *rdtype,
                                                 struct ompi_communicator_t *comm,
                                                 mca_coll_base_module_t *module)
{
    int ret;
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;

    if( !mca_coll_han_has_2_levels(han_module) ) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoall);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    ret = mca_coll_han_comm_create_new(comm, han_module);

    if( OMPI_SUCCESS != ret ) {  /* Let's hope the error is consistently returned across the entire communicator */
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                            "han cannot handle alltoall with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];

    /* Init topo */
    mca_coll_han_topo_init(comm, han_module, 2);

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoall with this communicator. "
                             "It need to fall back on another component\n"));
        return han_module->previous_alltoall(sbuf, scount, sdtype,
                                             rbuf, rcount, rdtype,
                                             comm, han_module->previous_alltoall_module);
    }

    /* Check if we have an igatherw for the up_comm */
    if (!up_comm->c_coll->coll_igatherw) {
        /* No igatherw for up_comm, use another algorithm */
        han_module->have_up_igatherw = false;
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, module);
    }

    /* MPI_IN_PLACE cases */
    const char *real_sbuf;
    char *alloc = NULL;
    int w_size = ompi_comm_size(comm);

    if (MPI_IN_PLACE == sbuf) {
        ptrdiff_t span, gap;
        span = opal_datatype_span(&rdtype->super, rcount * w_size, &gap);
        scount = rcount;
        sdtype = rdtype;
        alloc = (char*) malloc(span);
        ompi_datatype_copy_content_same_ddt(rdtype, rcount * w_size,
                                            alloc - gap, rbuf);
        real_sbuf = alloc - gap;
    } else {
        real_sbuf = sbuf;
    }

    /* Loop iterators */
    int step, up, low;

    /* Communicators infos */
    int low_rank, low_size, up_rank, up_size;
    int dest, node_root, up_root;

    /* Datatypes infos */
    ptrdiff_t sdtype_extent, rdtype_extent;
    size_t sdtype_size;

    /* Low communicator alltoallw informations */
    int *ata_scounts, *ata_sdispls;
    int *ata_rcounts, *ata_rdispls;
    struct ompi_datatype_t **ata_stypes, **ata_rtypes;

    /* Up communicator igatherw informations */
    int igw_scount;
    int *dt_displs;

    struct {
        int root;
        char *sbuf;
        int *rcounts, *rdispls;
        struct ompi_datatype_t **rtypes;
    } *igw;

    struct ompi_request_t **igw_rq;

    /* Get informations */
    low_size = ompi_comm_size(low_comm);
    up_size = ompi_comm_size(up_comm);
    low_rank = ompi_comm_rank(low_comm);
    up_rank = ompi_comm_rank(up_comm);

    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);
    ompi_datatype_type_size(sdtype, &sdtype_size);

    /* Allocate ressources */
    ata_scounts = (int*) malloc(low_size * sizeof(int));
    ata_rcounts = (int*) malloc(low_size * sizeof(int));
    ata_sdispls = (int*) malloc(low_size * sizeof(int));
    ata_rdispls = (int*) malloc(low_size * sizeof(int));
    ata_stypes = (struct ompi_datatype_t **) malloc(low_size * sizeof(struct ompi_datatype_t *));
    ata_rtypes = (struct ompi_datatype_t **) malloc(low_size * sizeof(struct ompi_datatype_t *));
    dt_displs = malloc(low_size * sizeof(int));
    igw = malloc(up_size * sizeof(*igw));
    igw_rq = malloc(up_size * sizeof(struct ompi_request_t *));

    /* Alltoall fixed values init */
    for (low = 0 ; low < low_size ; low++) {
        ata_scounts[low] = scount;
        ata_stypes[low] = sdtype;
        ata_rcounts[low] = scount * sdtype_size;
        ata_rtypes[low] = MPI_BYTE;
        ata_rdispls[low] = ata_rcounts[low] * low;
    }

    /* Build igatherw datatypes */
    igw_scount = low_size * sdtype_size * scount;
    for (step = 0 ; step < up_size ; step++) {
        /* Allocate intermediary buffer */
        igw[step].sbuf = malloc(igw_scount);

        igw_rq[step] = MPI_REQUEST_NULL;
        igw[step].root = (step + low_rank) % up_size;

        /* Only root needs to allocate igatherw infos */
        if (up_rank != igw[step].root) {
            igw[step].rcounts = NULL;
            igw[step].rdispls = NULL;
            igw[step].rtypes = NULL;
            continue;
        }

        igw[step].rcounts = malloc(up_size * sizeof(int));
        igw[step].rdispls = malloc(up_size * sizeof(int));
        igw[step].rtypes = malloc(up_size * sizeof(struct ompi_datatype_t *));

        for (up = 0 ; up < up_size ; up++) {
            node_root = han_module->global_ranks[up * han_module->maximum_size[LEAF_LEVEL]
                                                 + 0];
            igw[step].rdispls[up] = node_root * rcount * rdtype_extent;
            for (low = 0 ; low < low_size ; low++) {
                dest = han_module->global_ranks[up * han_module->maximum_size[LEAF_LEVEL]
                                                + low];
                dt_displs[low] = (dest - node_root) * rcount;
            }
            ompi_datatype_create_indexed_block(low_size,
                                               rcount,
                                               dt_displs,
                                               rdtype,
                                               &(igw[step].rtypes[up]));
            ompi_datatype_commit(&(igw[step].rtypes[up]));

            igw[step].rcounts[up] = 1;
        }
    }

    /* Perform communications */
    for (step = 0 ; step < up_size ; step++) {
        /* Compute displs for alltoallw sending buffer */
        for (low = 0 ; low < low_size ; low++) {
            /* up_root is the up rank of the root of the up igatherw
             * called by the rank which low_rank = low on our node
             */
            up_root = ((step + low) % up_size);
            dest = han_module->global_ranks[up_root * han_module->maximum_size[LEAF_LEVEL]
                                            + low];
            ata_sdispls[low] = dest * sdtype_extent * scount;
        }

        /* Perform alltoallw on intra node communicator */
        low_comm->c_coll->coll_alltoallw(real_sbuf,
                                         ata_scounts,
                                         ata_sdispls,
                                         ata_stypes,
                                         igw[step].sbuf,
                                         ata_rcounts,
                                         ata_rdispls,
                                         ata_rtypes,
                                         low_comm,
                                         low_comm->c_coll->coll_alltoallw_module);

        /* Initiate igatherw on inter node communicator */
        up_comm->c_coll->coll_igatherw(igw[step].sbuf,
                                       igw_scount,
                                       MPI_BYTE,
                                       rbuf,
                                       igw[step].rcounts,
                                       igw[step].rdispls,
                                       igw[step].rtypes,
                                       igw[step].root,
                                       up_comm,
                                       igw_rq + step,
                                       up_comm->c_coll->coll_igatherw_module);
    }

    /* Wait for completion */
    ompi_request_wait_all(up_size, igw_rq, MPI_STATUSES_IGNORE);

    /* Free ressources */
    for (step = 0 ; step < up_size ; step++) {
        if (up_rank == igw[step].root) {
            for (up = 0 ; up < up_size ; up++) {
                ompi_datatype_destroy(&(igw[step].rtypes[up]));
            }
            free(igw[step].rtypes);
            free(igw[step].rdispls);
            free(igw[step].rcounts);
        }
        free(igw[step].sbuf);
    }
    free(igw_rq);
    free(igw);
    free(dt_displs);
    free(ata_rtypes);
    free(ata_stypes);
    free(ata_rdispls);
    free(ata_sdispls);
    free(ata_rcounts);
    free(ata_scounts);

    if (NULL != alloc) {
        free(alloc);
    }

    return OMPI_SUCCESS;
}

/* Multi level alltoall grid algorithm :
 * There is an alltoall at each topological level and it start at leaf level if we're
 * in map_by_core situation, else we used alltoallw at leaf and top level and alltoall
 * for intermediate level.
 * Data are not directly send to destination, it will pass by intermediate rank
 * if there is no common sub_communication at specific topological level.
 * Example with 3 levels (Intra_node, Inter_node, Inter_cluster):
 * We are no in block distrisbution so alltoallw will be used.
 *       Cluster 0                Cluster 1
 * Node 0       Node 1      Node 2      Node 3
 *   0            1           3           5
 *
 *   2            4           6           7
 *
 * Focus on rank 0 state during the entire execution
 * Rank 0 buffer is: [00, 01, 02, 03, 04, 05, 06, 07]
 *
 * First step at leaf level:
 * Select all ranks with the same intra_node rank (leaf_level rank):
 * So rank 0 with rank 1, rank 3, and rank 5.
 * We create intermediary datatype to gather data for these ranks in a alltoallw:
 * Rank 0 buffer: [00, 01, 02, 03, 04, 05, 06, 07] => intra node alltoallw between rank 0 and rank 2
 * Rank 0 keep [00, 01, 03, 05] and send [02, 04, 06, 07]
 * Rank 2 keep [22, 24, 26, 27] and send [20, 21, 23, 25]
 * Rank 0 buffer after the first alltoall: [00, 01, 03, 05, 20, 21, 23, 25]
 *
 * Now we are in intermediate level we use alltoall because we know how data are ordered here.
 * Select all ranks with the same inter_node_rank:
 * Rank 0 with rank 3
 * Vector datatype is used here to select data to keep and send.
 * Rank 0 buffer: [00, 01, 03, 05, 20, 21, 23, 25] => inter node alltoall between rank 0 and rank 1
 * Rank 0 keep [00, 03, 20, 23] and send [01, 05, 21, 25]
 * Rank 1 keep [11, 15, 41, 45] and send [10, 13, 40, 43]
 * Rank 0 after alltoall: [00, 03, 20, 23, 10, 13, 40, 43]
 *
 * Last step, we use alltoallw to gather last data and reorder it in receive buffer
 * Vector is used to select data in send_buffer and another intermediary datatype in receive buffer to reorder data.
 * Rank 0 buffer: [00, 03, 20, 23, 10, 13, 40, 43] => inter cluster alltoallw between rank 0 and rank 3
 * Rank 0 keep [00, 20, 10, 40] and send [03, 23, 13, 43]
 * Rank 3 keep [33, 63, 53, 73] and send [30, 60, 50, 70]
 * Rank 0 before reordering in recvbuf [00, 20, 10, 40, 30, 60, 50, 70]
 * Rank 0 after alltoallw [00, 10, 20, 30, 40, 50, 60, 70]
 */

int mca_coll_han_ml_alltoall_grid(const void *sbuf, int scount,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module)
{
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int ret = mca_coll_han_comm_create_multi_level(comm, han_module);

    if( OMPI_SUCCESS != ret ) {  /* Let's hope the error is consistently returned across the entire communicator */
        opal_output_verbose(30, mca_coll_han_component.han_output,
                            "han failed to create sub_communicators. Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    /* unbalanced case needs algo adaptation */
    if (han_module->are_ppn_imbalanced) {
        opal_output_verbose(30, mca_coll_han_component.han_output,"%s cannot handle alltoall"
                            "with this communicator. Processes need to be perfectly balanced on each topological levels."
                            "Han  needs to fall back on another component to achieve Alltoall\n", __func__);
        return han_module->previous_alltoall(sbuf, scount, sdtype,
                                             rbuf, rcount, rdtype,
                                             comm, han_module->previous_alltoall_module);
    }
    /* If we are in block distribution, alltoall can be used at all level for better performance */
    if(han_module->is_mapbycore) {
        return mca_coll_han_ml_alltoall_grid_map_by_core(sbuf, scount, sdtype,
                                                         rbuf, rcount, rdtype,
                                                         comm, han_module);
    }
    /* Communicator info */
    int w_size = ompi_comm_size(comm);
    ompi_communicator_t **sub_comm = han_module->sub_comm;

    /* Use Temporary buffer during collective */
    void *buf_1 = NULL;
    void *buf_2 = NULL;
    const void *ata_sbuf;
    void *ata_rbuf;
    size_t type_size;
    ptrdiff_t sdtype_extent;
    ptrdiff_t rdtype_extent;
    int sub_comm_size;
    int ata_rcount;
    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);
    ompi_datatype_type_size(rdtype, &type_size);
    /* Size of one block */
    type_size = type_size * rcount;

    /* Search top leaf and top level */
    int top_comm_lvl;
    int start_comm_lvl = -1;
    int intermediate_comm_count = -2;
    for (int topo_lvl = 0; topo_lvl < GLOBAL_COMMUNICATOR; topo_lvl++) {
        if (ompi_comm_size(sub_comm[topo_lvl]) > 1) {
            top_comm_lvl = topo_lvl;
            intermediate_comm_count++;
            if (start_comm_lvl == -1) {
                start_comm_lvl = topo_lvl;
             }
        }
    }
    /* If there is no split fallback */
    if (start_comm_lvl == top_comm_lvl) {
        opal_output_verbose(30, mca_coll_han_component.han_output,"han cannot handle alltoall with this communicator,"
                            " there is only one topological level and han needs at least 2 levels."
                            "It need to fall back on another component\n");
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoall);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    /* Multiple buf allocation useful if there is at least 1 topologic levels between top and leaf level */
    buf_1 = malloc(type_size * w_size);
    if (intermediate_comm_count >= 1) {
        buf_2 = malloc(type_size * w_size);
    }

    /* First intermediate receive buffer */
    ata_rbuf = buf_1;
    if (MPI_IN_PLACE != sbuf) {
        ata_sbuf = sbuf;
    } else {
        ata_sbuf = rbuf;
        scount = rcount;
        sdtype = rdtype;
        sdtype_extent = rdtype_extent;
    }

    /* Save world rank topology, usefull for displacement and reorder */
    int *w_rank_topo;
    w_rank_topo = han_module->global_ranks;

    /* Leaf level Alltoallw init */
    int start_size = ompi_comm_size(sub_comm[start_comm_lvl]);
    int forwarded_rank_count = w_size / start_size;
    int *start_scount;
    int *start_rcount;
    int *start_sdisplacement;
    int *start_rdisplacement;
    struct ompi_datatype_t **start_sdtype;
    struct ompi_datatype_t **start_rdtype;

    start_scount = malloc(start_size * sizeof(int));
    start_rcount = malloc(start_size * sizeof(int));
    start_sdisplacement = malloc(start_size * sizeof(int));
    start_rdisplacement = malloc(start_size * sizeof(int));
    start_sdtype = malloc(start_size * sizeof(struct ompi_datatype_t *));
    start_rdtype = malloc(start_size * sizeof(struct ompi_datatype_t *));

    for (int sub_rank = 0; sub_rank < start_size; sub_rank++) {
        /* counts */
        start_scount[sub_rank] = 1;
        start_rcount[sub_rank] = forwarded_rank_count * (int)type_size;

        /* displacements */
        start_sdisplacement[sub_rank] = 0;
        start_rdisplacement[sub_rank] = sub_rank * forwarded_rank_count * (int)type_size;

        /* datatypes */
        start_sdtype[sub_rank] = ompi_datatype_create(forwarded_rank_count * scount * (2 + (int)sdtype->super.desc.used));
        /* create the datatype that contains the blocks for each remote that sub_rank peer will forward */
        for (int block = 0; block < forwarded_rank_count; block++) {
            /* Add the block of each peer which have the same sub_rank in their respective initial topology level (of size start_size) */
            ptrdiff_t remote_displ = sdtype_extent * w_rank_topo[start_size * block + sub_rank] * scount;
            ompi_datatype_add(start_sdtype[sub_rank], sdtype, scount, remote_displ, sdtype_extent);
        }
        ompi_datatype_commit(&(start_sdtype[sub_rank]));
        start_rdtype[sub_rank] = MPI_BYTE;

    }
    /* First alltoallw on leaf level to select which data will be sent and kept */
    sub_comm[start_comm_lvl]->c_coll->coll_alltoallw(ata_sbuf, start_scount, start_sdisplacement, start_sdtype,
                                                     ata_rbuf, start_rcount, start_rdisplacement, start_rdtype,
                                                     sub_comm[start_comm_lvl],
                                                     sub_comm[start_comm_lvl]->c_coll->coll_alltoallw_module);

    /* Free all buffer and datatypes from leaf_level alltoallw*/
    for(int size = 0; size<start_size; size++) {
        ompi_datatype_destroy(&(start_sdtype[size]));
    }
    free(start_scount);
    free(start_rcount);
    free(start_sdisplacement);
    free(start_rdisplacement);
    free(start_sdtype);
    free(start_rdtype);

    ata_sbuf = ata_rbuf;
    if (buf_2 != NULL) {
        ata_rbuf = buf_2;
    }
    /* After the first alltoallw all data are pack and ordered, we use classic alltoall on intermediate topo level */
    for (int topo_lvl = start_comm_lvl + 1; topo_lvl < top_comm_lvl; topo_lvl++) {
        sub_comm_size = ompi_comm_size(sub_comm[topo_lvl]);

        /* Always check if we are not own on our communicator */
        if (sub_comm_size <= 1) {
            continue;
        }
        ata_rcount = w_size / sub_comm_size * (int)type_size;
        struct ompi_datatype_t *vector_not_resized;
        struct ompi_datatype_t *vector_resized;

        ompi_datatype_create_vector(w_size / sub_comm_size, (int)type_size, sub_comm_size * (int)type_size, MPI_BYTE, &vector_not_resized);
        ompi_datatype_commit(&vector_not_resized);

        ompi_datatype_create_resized(vector_not_resized, 0, (int)type_size, &vector_resized);
        ompi_datatype_commit(&vector_resized);

        sub_comm[topo_lvl]->c_coll->coll_alltoall(ata_sbuf, 1, vector_resized,
                ata_rbuf, ata_rcount, MPI_BYTE,
                sub_comm[topo_lvl],
                sub_comm[topo_lvl]->c_coll->coll_alltoall_module);

        ata_sbuf = ata_rbuf;
        if(ata_rbuf != buf_1) {
            ata_rbuf = buf_1;
        } else {
            ata_rbuf = buf_2;
        }

        ompi_datatype_destroy(&vector_not_resized);
        ompi_datatype_destroy(&vector_resized);

    }
    /* last step with a reordering alltoallw */
    int top_size = ompi_comm_size(sub_comm[top_comm_lvl]);
    int *top_scount;
    int *top_rcount;
    int *top_sdisplacement;
    int *top_rdisplacement;
    struct ompi_datatype_t **top_sdtype;
    struct ompi_datatype_t **top_rdtype;

    top_scount = malloc(top_size * sizeof(int));
    top_rcount = malloc(top_size * sizeof(int));
    top_sdisplacement = malloc(top_size * sizeof(int));
    top_rdisplacement = malloc(top_size * sizeof(int));
    top_sdtype = malloc(top_size * sizeof(struct ompi_datatype_t *));
    top_rdtype = malloc(top_size * sizeof(struct ompi_datatype_t *));
    forwarded_rank_count = w_size / top_size;

    /* send datatype */
    struct ompi_datatype_t *vector;

    ompi_datatype_create_vector(forwarded_rank_count, (int)type_size, top_size * (int)type_size, MPI_BYTE, &vector);
    ompi_datatype_commit(&vector);

    for (int sub_rank = 0; sub_rank < top_size; sub_rank++) {
        /* counts */
        top_scount[sub_rank] = 1;
        top_rcount[sub_rank] = 1;

        /* displacements */
        top_sdisplacement[sub_rank] = (int)type_size * sub_rank;
        top_rdisplacement[sub_rank] = 0;

        /* datatypes */
        top_sdtype[sub_rank] = vector;
        top_rdtype[sub_rank] = ompi_datatype_create(forwarded_rank_count * rcount * (2 + (int)rdtype->super.desc.used));
        /* create the datatype that contains the reordered block for the rbuf*/
        for (int block = 0; block < forwarded_rank_count; block++) {
            /* Add the block of each peer which have the same sub_rank in their respective topology level (of size top_size) */
            ptrdiff_t remote_displ = rdtype_extent * w_rank_topo[forwarded_rank_count * sub_rank + block] * rcount;
            ompi_datatype_add(top_rdtype[sub_rank], rdtype, rcount, remote_displ, rdtype_extent);
        }
        ompi_datatype_commit(&(top_rdtype[sub_rank]));
    }

    sub_comm[top_comm_lvl]->c_coll->coll_alltoallw(ata_sbuf, top_scount, top_sdisplacement, top_sdtype,
                                                   rbuf, top_rcount, top_rdisplacement, top_rdtype,
                                                   sub_comm[top_comm_lvl],
                                                   sub_comm[top_comm_lvl]->c_coll->coll_alltoallw_module);

    /* Free all buffer and datatypes from top_level alltoallw*/
    ompi_datatype_destroy(&vector);
    for(int size = 0; size<top_size; size++) {
        ompi_datatype_destroy(&(top_rdtype[size]));
    }
    free(top_scount);
    free(top_rcount);
    free(top_sdisplacement);
    free(top_rdisplacement);
    free(top_sdtype);
    free(top_rdtype);

    free(buf_1);
    free(buf_2);

    return OMPI_SUCCESS;
}

/* Grid map_by_core version */
static int mca_coll_han_ml_alltoall_grid_map_by_core(const void *sbuf, int scount,
                                                     struct ompi_datatype_t *sdtype,
                                                     void *rbuf, int rcount,
                                                     struct ompi_datatype_t *rdtype,
                                                     struct ompi_communicator_t *comm,
                                                     mca_coll_han_module_t *han_module)
{
    int w_size = ompi_comm_size(comm);
    ompi_communicator_t **sub_comm = han_module->sub_comm;

    /* Use Temporary buffer during collective */
    void *buf_1 = NULL;
    void *buf_2 = NULL;
    const void *ata_sbuf;
    void *ata_rbuf;
    size_t type_size;
    ptrdiff_t sdtype_extent;
    ptrdiff_t rdtype_extent;
    int sub_comm_size;
    int ata_rcount;
    struct ompi_datatype_t *ata_rdtype;

    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);
    ompi_datatype_type_size(rdtype, &type_size);
    type_size = rcount * type_size;

    /* Search top leaf and top level */
    int top_comm_lvl;
    int start_comm_lvl = -1;
    int intermediate_comm_count = -2;
    for (int topo_lvl = 0; topo_lvl < GLOBAL_COMMUNICATOR; topo_lvl++) {
        if (ompi_comm_size(sub_comm[topo_lvl]) > 1) {
            top_comm_lvl = topo_lvl;
            intermediate_comm_count++;
            if (start_comm_lvl == -1) {
                start_comm_lvl = topo_lvl;
             }
        }
    }

    /* If there is no split fallback */
    if (start_comm_lvl == top_comm_lvl) {
        opal_output_verbose(30, mca_coll_han_component.han_output,"han cannot handle alltoall with this communicator,"
                            " there is only one topological level and han needs at least 2 levels."
                            "It need to fall back on another component\n");
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoall);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    /* Multiple buf allocation useful if there is at least 1 topologic levels between top and leaf level */
    buf_1 = malloc(type_size * w_size);
    if (intermediate_comm_count >= 1) {
        buf_2 = malloc(type_size * w_size);
    }

    if (MPI_IN_PLACE != sbuf) {
        ata_sbuf = sbuf;
    } else {
        ata_sbuf = rbuf;
        scount = rcount;
        sdtype = rdtype;
        sdtype_extent = rdtype_extent;
    }

    ata_rbuf = buf_1;
    ata_rdtype = MPI_BYTE;
    for (int topo_lvl = start_comm_lvl; topo_lvl <= top_comm_lvl; topo_lvl++) {
        sub_comm_size = ompi_comm_size(sub_comm[topo_lvl]);

        if(sub_comm_size <= 1) {
            continue;
        }
        if (topo_lvl == top_comm_lvl) {
            ata_rbuf = rbuf;
            ata_rdtype = rdtype;
            ata_rcount = w_size / sub_comm_size * rcount;
        } else {
             ata_rcount = w_size / sub_comm_size * (int)type_size;
        }
        struct ompi_datatype_t *vector_not_resized;
        struct ompi_datatype_t *vector_resized;

        if (topo_lvl == start_comm_lvl) {
            ompi_datatype_create_vector(w_size / sub_comm_size, scount, sub_comm_size * scount, sdtype, &vector_not_resized);
            ompi_datatype_commit(&vector_not_resized);

            ompi_datatype_create_resized(vector_not_resized, 0, sdtype_extent  * scount, &vector_resized);
            ompi_datatype_commit(&vector_resized);
        } else {
            ompi_datatype_create_vector(w_size / sub_comm_size, (int)type_size, sub_comm_size * (int)type_size, MPI_BYTE, &vector_not_resized);
            ompi_datatype_commit(&vector_not_resized);

            ompi_datatype_create_resized(vector_not_resized, 0, (int)type_size, &vector_resized);
            ompi_datatype_commit(&vector_resized);
        }

        sub_comm[topo_lvl]->c_coll->coll_alltoall(ata_sbuf, 1, vector_resized,
                ata_rbuf, ata_rcount, ata_rdtype,
                sub_comm[topo_lvl],
                sub_comm[topo_lvl]->c_coll->coll_alltoall_module);

        ata_sbuf = ata_rbuf;
        if(ata_rbuf != buf_1) {
            ata_rbuf = buf_1;
        } else {
            ata_rbuf = buf_2;

        }
        ompi_datatype_destroy(&vector_not_resized);
        ompi_datatype_destroy(&vector_resized);
    }

    free(buf_1);
    free(buf_2);

    return OMPI_SUCCESS;
}


/* Macro: tests if a position is in a sub_tree */
#define POS_IN_TREE_NODE(rank_pos, tree_node) \
    ((rank_pos) >= (tree_node)->wranks_range.start \
     && (rank_pos) < (tree_node)->wranks_range.start \
                     + (tree_node)->wranks_range.nb)
/* mca_coll_han_ml_alltoall_gas0 utilitary function
 * returns the position of the buffer from source to target on topo_lvl gather
 * returns -1 if the buffer from source to target is not handled on this topo_lvl
 */
static int
buffer_position_gather(int source, int target, int topo_lvl,
                       struct ompi_communicator_t *comm,
                       const mca_coll_han_module_t *han_module)
{
    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Actual gather topo tree node */
    const mca_coll_han_topo_tree_node_t *topo_node = topo_tree->my_sub_tree[topo_lvl];

    /* Previous gather topo tree node */
    mca_coll_han_topo_tree_node_t *previous_node = topo_tree->my_sub_tree[topo_lvl-1];

    /* Where source and target are in the rank_from_topo_index table */
    int source_pos = topo_tree->topo_index_from_rank[source];
    int target_pos = topo_tree->topo_index_from_rank[target];

    /* Shortcut: no need to send this buffer if:
     *    - the target is in previous tree
     *    - the source is not in previous tree
     */
    if (POS_IN_TREE_NODE(target_pos, previous_node)
        || !POS_IN_TREE_NODE(source_pos, topo_node)) {
        return -1;
    }

    int sub_tree_offset = 0;
    int source_offset = 0;
    int target_offset = 0;

    /* If we skiped the previous gather, dive deeper */
    while (previous_node->n_sub_tree == 1) {
        previous_node = previous_node->sub_tree;
    }

    /* In case of leaf_level (n_sub_tree = 0) */
    mca_coll_han_topo_tree_node_t *sub_tree = previous_node;
    for (int sub_tree_id = 0; sub_tree_id < previous_node->n_sub_tree; sub_tree_id++) {
        sub_tree = &previous_node->sub_tree[sub_tree_id];

        /* We skip MPI_COMM_SELF levels, dive deeper */
        while (1 == sub_tree->n_sub_tree) {
            sub_tree = sub_tree->sub_tree;
        }

        /* Source is in this sub_tree */
        if (POS_IN_TREE_NODE(source_pos, sub_tree)) {
            break;
        }

        /* Source is not in this sub_tree
         * Add buffer displ
         * The remaining data from each of the wranks_range.nb ranks in this sub tree
         * are data that target ranks outside of this sub tree
         * So there are wranks_range.nb ranks * (comm_size - wranks_range.nb) blocks
         */
        sub_tree_offset += sub_tree->wranks_range.nb
                           * (ompi_comm_size(comm) - sub_tree->wranks_range.nb);
    }

    /* Number of blocks sent by sources before this source */
    if (0 == previous_node->n_sub_tree) {
        /* LEAF_LEVEL case */
        source_offset = (source_pos - sub_tree->wranks_range.start)
                        * (ompi_comm_size(comm) - 1);
    } else {
        source_offset = (source_pos - sub_tree->wranks_range.start)
                        * (ompi_comm_size(comm) - sub_tree->wranks_range.nb);
    }

    if (target_pos < sub_tree->wranks_range.start) {
        /* This target is before the hole */
        target_offset = target_pos;
    } else if (0 == previous_node->n_sub_tree) {
        /* LEAF_LEVEL case */
        target_offset = target_pos - 1;
    } else {
        /* This target is after the hole
         * Remove the size of the hole
         */
        target_offset = target_pos - sub_tree->wranks_range.nb;
    }

    return sub_tree_offset + source_offset + target_offset;
}

/* mca_coll_han_ml_alltoall_gas0 utilitary function
 * Check if the buffer should be treated for this scatter and where the buffer is
 * P = MY_PARENT = taken from recvbuf of previous scatter/alltoall
 * M = MYSELF = locked during ascending gather in this topo_lvl
 * I = IDONTCARE = locked before or do not use it on this topo_lvl
 *
 *                 /\
 *                / I\
 *               /<---\  __me
 *              /  P   \/
 *             /------>/\<----topo_lvl
 *            /       / M\
 *           /       /<-->\
 *          /\      /\     \
 *         / I\    / I\     \
 *        /<-->\  /<-->\     \
 */
enum data_provider { MY_PARENT, MYSELF, IDONTCARE};
static enum data_provider
find_descending_data_owner(int source, int target, int topo_lvl, const mca_coll_han_module_t *han_module)
{
    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Actual scatter topo tree node */
    const mca_coll_han_topo_tree_node_t *topo_node = topo_tree->my_sub_tree[topo_lvl];

    /* Source and target position in the rank_from_topo_index table */
    int source_pos = topo_tree->topo_index_from_rank[source];
    int target_pos = topo_tree->topo_index_from_rank[target];

    if (!POS_IN_TREE_NODE(target_pos, topo_node)) {
        return IDONTCARE;
    }

    if (!POS_IN_TREE_NODE(source_pos, topo_node)) {
        return MY_PARENT;
    }

    /* Shortcut for LEAF_LEVEL */
    if (LEAF_LEVEL == topo_lvl) {
        if (source != target) {
            return MYSELF;
        } else {
            return IDONTCARE;
        }
    }

    /* Find the source sub_tree */
    int left = 0;
    int middle;
    int right = topo_node->n_sub_tree-1;

    /* Find the sub_tree the source is in (dichotomy) */
    while (left+1 < right) {
        middle = (left + right) / 2;
        if (source_pos < topo_node->sub_tree[middle].wranks_range.start) {
            right = middle;
        } else {
            left = middle;
        }
    }

    /* Dichotomy ending */
    int source_tree = 0;
    if (source_pos < topo_node->sub_tree[right].wranks_range.start) {
        source_tree = left;
    } else {
        source_tree = right;
    }

    /* Check if target is in the same sub_tree */
    if (POS_IN_TREE_NODE(target_pos, &topo_node->sub_tree[source_tree])) {
        /* Same sub_tree */
        return IDONTCARE;
    } else {
        /* Different sub_tree */
        return MYSELF;
    }
}

static int
buffer_position_scatter_from_alltoall(int source, int target, int topo_lvl,
                                      const struct ompi_communicator_t *comm,
                                      const mca_coll_han_module_t *han_module)
{
    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Actual scatter topo tree node */
    const mca_coll_han_topo_tree_node_t *topo_node = topo_tree->my_sub_tree[topo_lvl];

    /* Where source and target are in the rank_from_topo_index table */
    int source_pos = topo_tree->topo_index_from_rank[source];
    int target_pos = topo_tree->topo_index_from_rank[target];

    int buffer_pos = 0;

    /* First scatter: the previous level was an alltoall */
    int source_sub_tree_offset = 0;
    int source_offset = 0;
    int target_offset = 0;

    const mca_coll_han_topo_tree_node_t *global_tree_node = topo_tree->my_sub_tree[GLOBAL_COMMUNICATOR-1];

    /* Find the sub_tree the source is in */
    for (int sub_tree_id = 0; sub_tree_id < global_tree_node->n_sub_tree; sub_tree_id++) {
        const mca_coll_han_topo_tree_node_t *sub_tree_node = &global_tree_node->sub_tree[sub_tree_id];

        /* We do not send anything to ourself during alltoall */
        if (POS_IN_TREE_NODE(target_pos, sub_tree_node)) {
            continue;
        }

        /* Is source in this sub_tree? */
        if (POS_IN_TREE_NODE(source_pos, sub_tree_node)) {
            /* Data is sorted by target then by source */
            target_offset = (target_pos - topo_node->wranks_range.start)
                            * sub_tree_node->wranks_range.nb;

            source_offset = source_pos - sub_tree_node->wranks_range.start;
            break;
        }

        /* Add offset: data sent from this sub_tree to our sub_tree */
        source_sub_tree_offset += topo_node->wranks_range.nb * sub_tree_node->wranks_range.nb;
    }

    buffer_pos = source_sub_tree_offset + source_offset + target_offset;

    return buffer_pos;
}

static int
buffer_position_scatter_from_scatter(int source, int target, int topo_lvl,
                                     struct ompi_communicator_t *comm,
                                     const mca_coll_han_module_t *han_module)
{
    int comm_size = ompi_comm_size(comm);

    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Actual scatter topo tree node */
    const mca_coll_han_topo_tree_node_t *topo_node = topo_tree->my_sub_tree[topo_lvl];

    /* Where source and target are in the rank_from_topo_index table */
    int source_pos = topo_tree->topo_index_from_rank[source];
    int target_pos = topo_tree->topo_index_from_rank[target];

    int buffer_pos = 0;

    int source_offset;
    int target_offset;

    /* Data is sorted by target then by source */
    target_offset = (target_pos - topo_node->wranks_range.start)
                    * (comm_size - topo_node->wranks_range.nb);

    if (source_pos < topo_node->wranks_range.start) {
        /* Before the hole */
        source_offset = source_pos;
    } else {
        /* After the hole */
        source_offset = source_pos - topo_node->wranks_range.nb;
    }

    buffer_pos = source_offset + target_offset;

    return buffer_pos;
}

static int
buffer_position_scatter_from_gather(int source, int target, int topo_lvl,
                                    struct ompi_communicator_t *comm,
                                    const mca_coll_han_module_t *han_module)
{
    int comm_size = ompi_comm_size(comm);

    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Actual scatter topo tree node */
    const mca_coll_han_topo_tree_node_t *topo_node = topo_tree->my_sub_tree[topo_lvl];

    /* Where source and target are in the rank_from_topo_index table */
    int source_pos = topo_tree->topo_index_from_rank[source];
    int target_pos = topo_tree->topo_index_from_rank[target];

    int buffer_pos = 0;

    int source_offset = 0;
    int target_offset = 0;

    if (LEAF_LEVEL == topo_lvl) {
        /* Data is sorted by source then by target */
        source_offset = (comm_size - 1) * (source_pos - topo_node->wranks_range.start);

        if (target_pos < source_pos) {
            target_offset = target_pos;
        } else {
            target_offset = target_pos-1;
        }
    } else {
        const mca_coll_han_topo_tree_node_t *sub_tree_node = topo_node->sub_tree;

        /* Find the sub_tree the source is in
         * n_sub_tree > 0 if we are not in LEAF_LEVEL
         */
        for (int sub_tree_id = 0; sub_tree_id < topo_node->n_sub_tree; sub_tree_id++) {
            sub_tree_node = &topo_node->sub_tree[sub_tree_id];

            if (POS_IN_TREE_NODE(source_pos, sub_tree_node)) {
                /* Source is in this sub_tree
                 * Add messages before source in this sub_tree
                 */
                source_offset += (comm_size - sub_tree_node->wranks_range.nb)
                                 * (source_pos - sub_tree_node->wranks_range.start);
                break;
            }

            /* Source is not in this sub_tree, add messages from this sub_tree */
            source_offset += (comm_size - sub_tree_node->wranks_range.nb)
                             * sub_tree_node->wranks_range.nb;
        }

        if (target_pos < sub_tree_node->wranks_range.start) {
            target_offset = target_pos;
        } else {
            target_offset = target_pos - sub_tree_node->wranks_range.nb;
        }
    }

    buffer_pos = source_offset + target_offset;

    return buffer_pos;
}

/* mca_coll_han_ml_alltoall_gas0 utilitary function
 * returns the position of the buffer from source to target on topo_lvl scatter
 * returns -1 if the buffer from source to target is not handled on this topo_lvl
 *
 * used with the find_descending_data_owner function to identify which buffer to use
 */
static int
buffer_position_scatter(int source, int target, int topo_lvl,
                        struct ompi_communicator_t *comm, const mca_coll_han_module_t *han_module,
                        enum data_provider *buffer_owner)
{
    *buffer_owner = find_descending_data_owner(source, target, topo_lvl, han_module);

    /* -1 if this buffer is not handled in this scatter */
    if (IDONTCARE == *buffer_owner) {
        return -1;
    }

    /* Alias */
    const mca_coll_han_topo_tree_t *topo_tree = han_module->topo_tree;

    /* Buffer is in alltoall/previous scatter rbuf */
    if (MY_PARENT == *buffer_owner) {
        int previous_topo_lvl = topo_lvl+1;

        /* If previous level was a self level, climb higher */
        while (previous_topo_lvl < GLOBAL_COMMUNICATOR-1
               && 1 == topo_tree->my_sub_tree[previous_topo_lvl]->n_sub_tree) {
            previous_topo_lvl++;
        }

        if (GLOBAL_COMMUNICATOR-1 == previous_topo_lvl) {
            return buffer_position_scatter_from_alltoall(source, target, topo_lvl, comm, han_module);
        } else {
            return buffer_position_scatter_from_scatter(source, target, topo_lvl, comm, han_module);
        }
    }

    /* Buffer blocked during gather loop */
    if (MYSELF == *buffer_owner) {
        return buffer_position_scatter_from_gather(source, target, topo_lvl, comm, han_module);
    }
}

/* Multi level alltoall algorithm
 * This algorithm is based on the topology tree representation.
 * Here is an example of an imbalanced 3-leveled topology tree:
 *
 * Inter node      ________________________
 *                /          |             \
 *               0           2              4
 *               |           |              |
 * Inter socket  |____       |____          |
 *               |    \      |    \         |
 *               0     1     2     6        4
 *               |     |     |     |        |
 * Leaf level    |_    |_    |_    |____    |
 *               | \   | \   | \   | \  \   |
 *               0  3  1  5  2  8  6  7  9  4
 *
 * This algorithm use gathers climbing the topology tree, an alltoall at top lvl
 * and scatters diving the tree back.
 *
 * The purpose of this algorithm is to have the least process and data to manage for
 * the highest topological levels.
 *
 * Only the roots of the previous gather participate in the next gather.
 * Messages only climb the topology tree to reach their target.
 * Once the target can be reach diving the topology tree,
 * messages are blocked and are not packed for gather anymore.
 * For example, message from 7 to 2 does not need
 * to climb higher than inter socket level.
 * During the gather loop, messages are sorted by source then by target.
 *
 * An alltoall is performed on the top level on the last gather roots.
 * Each rank involved in the alltoall receives the messages destinated to
 * its part of the topology tree.
 *
 * After the alltoall, messages are sorted by target then by source.
 *
 * Message then dive the tree through a scatter loop.
 * Messages blocked during gathers loop are packed
 * with descending messages during scatters loop.
 *
 */
int mca_coll_han_ml_alltoall_gas0(const void *sbuf, int scount,
                                  struct ompi_datatype_t *sdtype,
                                  void *rbuf, int rcount,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  mca_coll_base_module_t *module)
{
    /* Create the subcommunicators */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    int ret = mca_coll_han_comm_create_multi_level(comm, han_module);

    if (OMPI_SUCCESS != ret) {  /* Let's hope the error is consistently returned across the entire communicator */
        opal_output_verbose(30, mca_coll_han_component.han_output,
                            "han failed to create sub_communicators. Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoall(sbuf, scount, sdtype,
                                           rbuf, rcount, rdtype,
                                           comm, han_module->previous_alltoall_module);
    }

    /* Init */
    int world_size = ompi_comm_size(comm);
    int world_rank = ompi_comm_rank(comm);
    int my_sub_rank;
    int sub_comm_size;
    ompi_communicator_t **sub_comm = han_module->sub_comm;

    /* All collective will use MPI_BYTE so we need to store type size */
    size_t type_size;
    ompi_datatype_type_size(rdtype, &type_size);
    const mca_coll_han_topo_tree_t *topo = han_module->topo_tree;
    int topo_lvl;
    int higher_gather_lvl;
    int last_topo_lvl;

    /* Climbing gather loop */

    /* Allocate multiple buffer for each steps of gatherv*/
    void *gatherv_send_buf[GLOBAL_COMMUNICATOR-1];
    void *gatherv_recv_buf[GLOBAL_COMMUNICATOR-1];
    void *gatherv_copy_buf[GLOBAL_COMMUNICATOR-1];
    int gatherv_send_count[GLOBAL_COMMUNICATOR-1];
    int *gatherv_recv_count[GLOBAL_COMMUNICATOR-1];
    int *gatherv_recv_displ[GLOBAL_COMMUNICATOR-1];

    /* Initialise*/
    for (topo_lvl = 0; topo_lvl < GLOBAL_COMMUNICATOR-1; topo_lvl++) {
        gatherv_send_buf[topo_lvl] = NULL;
        gatherv_recv_buf[topo_lvl] = NULL;
        gatherv_copy_buf[topo_lvl] = NULL;
        gatherv_send_count[topo_lvl] = 0;
        gatherv_recv_count[topo_lvl] = NULL;
        gatherv_recv_displ[topo_lvl] = NULL;
    }

    /* First gatherv at leaf level where every process participate*/
    /* Root check */
    if (0 == ompi_comm_rank(sub_comm[LEAF_LEVEL])) {
        sub_comm_size = ompi_comm_size(sub_comm[LEAF_LEVEL]);
        /* We left our data in send buff so size is world_size - 1 */
        gatherv_recv_buf[LEAF_LEVEL] = malloc(type_size * rcount * (world_size - 1) * sub_comm_size);
        /* Root copies its data directly into the recv buff */
        gatherv_copy_buf[LEAF_LEVEL] = gatherv_recv_buf[LEAF_LEVEL];
        gatherv_recv_count[LEAF_LEVEL] = malloc(sizeof(int) * sub_comm_size);
        gatherv_recv_displ[LEAF_LEVEL] = malloc(sizeof(int) * sub_comm_size);
        if (NULL == gatherv_recv_count[LEAF_LEVEL]
            || NULL == gatherv_recv_displ[LEAF_LEVEL]
            || NULL == gatherv_recv_buf[LEAF_LEVEL]) {
            goto cleanup_and_fallback_gather;
        }
        gatherv_recv_count[LEAF_LEVEL][0] = 0;
        gatherv_recv_displ[LEAF_LEVEL][0] = 0;
        for (int leaf_rank = 1; leaf_rank < sub_comm_size; leaf_rank++) {
           gatherv_recv_count[LEAF_LEVEL][leaf_rank] = (world_size - 1) * type_size * rcount;
           gatherv_recv_displ[LEAF_LEVEL][leaf_rank] = (world_size - 1) * type_size * rcount * leaf_rank;
        }
    } else {
        gatherv_send_buf[LEAF_LEVEL] =  malloc(type_size * rcount * (world_size - 1));
        if (NULL == gatherv_send_buf[LEAF_LEVEL]) {
            goto cleanup_and_fallback_gather;
        }
        gatherv_send_count[LEAF_LEVEL] = rcount * type_size * (world_size - 1);
        gatherv_copy_buf[LEAF_LEVEL] = gatherv_send_buf[LEAF_LEVEL];
    }

    /* Data is pack so as not to leave any hole in buffer and it is reordered in
     * the rank_from_topo_index table order */
    ptrdiff_t sextent;
    ompi_datatype_type_extent(sdtype, &sextent);
    ptrdiff_t block_size = sextent * (ptrdiff_t)scount;
    int recv_position = 0;
    for (int source_pos = 0; source_pos < world_size; source_pos++) {
        if (topo->rank_from_topo_index[source_pos] != world_rank) {
            ptrdiff_t src_shift = block_size * topo->rank_from_topo_index[source_pos];
            ptrdiff_t dest_shift = rcount * type_size * recv_position;
            recv_position++;
            /* Convert sdtype into MPI_BYTE */
            ompi_datatype_sndrcv((char *)sbuf + src_shift, scount, sdtype,
                                 (char *)gatherv_copy_buf[LEAF_LEVEL] + dest_shift, rcount * type_size, MPI_BYTE);
        }
    }
    sub_comm[LEAF_LEVEL]->c_coll->coll_gatherv(gatherv_send_buf[LEAF_LEVEL],
                                               gatherv_send_count[LEAF_LEVEL],
                                               MPI_BYTE,
                                               gatherv_recv_buf[LEAF_LEVEL],
                                               gatherv_recv_count[LEAF_LEVEL],
                                               gatherv_recv_displ[LEAF_LEVEL],
                                               MPI_BYTE,
                                               0, sub_comm[LEAF_LEVEL],
                                               sub_comm[LEAF_LEVEL]->c_coll->coll_gatherv_module);

    /* From now on, data is stored as MPI_BYTE */
    block_size = type_size * (ptrdiff_t)rcount;

    higher_gather_lvl = LEAF_LEVEL;
    last_topo_lvl = LEAF_LEVEL;
    /* Climbing loop of gatherv, finish just before the last level */
    for (topo_lvl = LEAF_LEVEL + 1; topo_lvl < GLOBAL_COMMUNICATOR - 1; topo_lvl++) {
        sub_comm_size = ompi_comm_size(sub_comm[topo_lvl]);
        /* Only previous root can participate */
        if (0 < ompi_comm_rank(sub_comm[last_topo_lvl])) {
            break;
        }
        /* If process is alone on a sub_comm, it go to the next level*/
        if (1 == sub_comm_size) {
            continue;
        }

        const mca_coll_han_topo_tree_node_t * last_topo =  topo->my_sub_tree[last_topo_lvl];

        const mca_coll_han_topo_tree_node_t *topo_node = topo->my_sub_tree[topo_lvl];
        /* Shift created by data that is copied directly into the recv buffer by the root */
        ptrdiff_t root_displ = (world_size - topo_node->sub_tree[0].wranks_range.nb)
                               * topo_node->sub_tree[0].wranks_range.nb
                               * type_size
                               * rcount;
        /* Root init */
        if (0 == ompi_comm_rank(sub_comm[topo_lvl])) {
             int nb_block = 0;
             /* compute size of recv buff by checking out topo struct */
             for (int sub_tree_id = 0; sub_tree_id < topo_node->n_sub_tree; sub_tree_id++) {
                 int sub_tree_size = topo_node->sub_tree[sub_tree_id].wranks_range.nb;
                 nb_block += (world_size - sub_tree_size) * sub_tree_size;
             }
             gatherv_recv_buf[topo_lvl] = malloc(type_size * rcount * nb_block);
             gatherv_copy_buf[topo_lvl] = gatherv_recv_buf[topo_lvl];
             gatherv_recv_count[topo_lvl] = malloc(sizeof(int) * sub_comm_size);
             gatherv_recv_displ[topo_lvl] = malloc(sizeof(int) * sub_comm_size);
             if (NULL == gatherv_recv_buf[topo_lvl]
                 || NULL == gatherv_recv_count[topo_lvl]
                 || NULL == gatherv_recv_displ[topo_lvl]) {
                goto cleanup_and_fallback_gather;
             }
             gatherv_recv_count[topo_lvl][0] = 0;
             gatherv_recv_displ[topo_lvl][0] = 0;
             for (int sub_comm_rank = 1; sub_comm_rank < sub_comm_size; sub_comm_rank++) {
                 gatherv_recv_count[topo_lvl][sub_comm_rank] = (world_size - topo_node->sub_tree[sub_comm_rank].wranks_range.nb)
                                                               * topo_node->sub_tree[sub_comm_rank].wranks_range.nb
                                                               * type_size * rcount;
                 gatherv_recv_displ[topo_lvl][sub_comm_rank] = gatherv_recv_count[topo_lvl][sub_comm_rank-1]
                                                               + gatherv_recv_displ[topo_lvl][sub_comm_rank-1];
             }
        } else {
            gatherv_send_buf[topo_lvl] =  malloc(type_size * rcount
                                                 * (world_size - last_topo->wranks_range.nb)
                                                 * last_topo->wranks_range.nb);
            if(NULL == gatherv_send_buf[topo_lvl]) {
                goto cleanup_and_fallback_gather;
            }
            gatherv_send_count[topo_lvl] = type_size * rcount
                                           * (world_size - last_topo->wranks_range.nb)
                                           * last_topo->wranks_range.nb;
            gatherv_copy_buf[topo_lvl] = gatherv_send_buf[topo_lvl];
        }

        /* Pack all data to not left any hole */
        ptrdiff_t gather_block_size = type_size * (ptrdiff_t)rcount;
        recv_position = 0;

        /* Loop on all possible process data source */
        for (int topo_index = 0; topo_index < last_topo->wranks_range.nb; topo_index++) {

            /* Process data source */
            int source = topo->rank_from_topo_index[last_topo->wranks_range.start + topo_index];

            /* Loop on all possible process data destination */
            for (int block = 0; block < world_size; block++) {
                /* Process data destination in rank_from_topo_index table order*/
                int dest = topo->rank_from_topo_index[block];

                /* Find data position in previous gather recv buffer */
                int send_position = buffer_position_gather(source, dest, topo_lvl, comm, han_module);

                /* If data have to be keep, we left its in the previous recv buffer */
                if (-1 != send_position) {
                    ptrdiff_t src_shift = gather_block_size * send_position;
                    ptrdiff_t dest_shift = gather_block_size * recv_position;
                    recv_position++;

                    /* Pack this buffer */
                    ompi_datatype_sndrcv((char *)gatherv_recv_buf[last_topo_lvl] + src_shift,
                                         rcount * type_size, MPI_BYTE,
                                         (char *)gatherv_copy_buf[topo_lvl] + dest_shift,
                                         rcount * type_size, MPI_BYTE);
                }
            }
        }
        sub_comm[topo_lvl]->c_coll->coll_gatherv(gatherv_send_buf[topo_lvl],
                                                 gatherv_send_count[topo_lvl],
                                                 MPI_BYTE,
                                                 gatherv_recv_buf[topo_lvl] + root_displ,
                                                 gatherv_recv_count[topo_lvl],
                                                 gatherv_recv_displ[topo_lvl],
                                                 MPI_BYTE,
                                                 0, sub_comm[topo_lvl],
                                                 sub_comm[topo_lvl]->c_coll->coll_gatherv_module);

        /* Update process last participation */
        higher_gather_lvl = topo_lvl;
        last_topo_lvl = topo_lvl;
    }

    /* Alltoallv at top level */
    void *alltoallv_recv_buf=NULL;
    int was_in_alltoall = false;

    /* Only the root of last gatherv participate in the alltoallv */
    if (0 == ompi_comm_rank(sub_comm[last_topo_lvl])) {
        was_in_alltoall = true;
        sub_comm_size = ompi_comm_size(sub_comm[topo_lvl]);
        my_sub_rank = ompi_comm_rank(sub_comm[topo_lvl]);

        /* Alltoallv init */
        void *alltoallv_send_buf=NULL;
        int alltoallv_count[sub_comm_size];
        int alltoallv_displ[sub_comm_size];
        const mca_coll_han_topo_tree_node_t *alltoallv_topo_node = topo->my_sub_tree[topo_lvl];
        int alltoallv_nb_block = (world_size - alltoallv_topo_node->sub_tree[my_sub_rank].wranks_range.nb)
                                 * alltoallv_topo_node->sub_tree[my_sub_rank].wranks_range.nb;

        /* Same count of data send and recv */
        alltoallv_send_buf = malloc(type_size * rcount * alltoallv_nb_block);
        alltoallv_recv_buf = malloc(type_size * rcount * alltoallv_nb_block);
        if (NULL == alltoallv_send_buf || NULL == alltoallv_recv_buf) {
            if (NULL != alltoallv_send_buf) {
                free(alltoallv_send_buf);
            }
            goto cleanup_and_fallback_alltoall;
        }
        for (int sub_comm_rank = 0; sub_comm_rank < sub_comm_size; sub_comm_rank++) {
            /* All data needed for my process and for the other processes sharing my
             * top_level have already been left in all previous gather recv buffers */
            if(my_sub_rank == sub_comm_rank) {
                alltoallv_count[sub_comm_rank] = 0;
            } else {
                alltoallv_count[sub_comm_rank] = alltoallv_topo_node->sub_tree[my_sub_rank].wranks_range.nb
                                                 * alltoallv_topo_node->sub_tree[sub_comm_rank].wranks_range.nb
                                                 * rcount * type_size;
            }
            if(0 == sub_comm_rank)
            {
                alltoallv_displ[0] = 0;
            } else {
                alltoallv_displ[sub_comm_rank] = alltoallv_count[sub_comm_rank-1]
                                                 + alltoallv_displ[sub_comm_rank-1];
            }
        }

        /*Data reorder in destination order and pack in send buffer*/
        recv_position = 0;

        const mca_coll_han_topo_tree_node_t * last_topo =  topo->my_sub_tree[last_topo_lvl];
        /* Loop on process data destination */
        for (int dest_pos = 0; dest_pos < world_size; dest_pos++) {
            int dest = topo->rank_from_topo_index[dest_pos];

            /* Loop on process data source */
            for (int sub_comm_rank = 0; sub_comm_rank < last_topo->wranks_range.nb; sub_comm_rank++) {
                int source = topo->rank_from_topo_index[last_topo->wranks_range.start + sub_comm_rank];
                int send_position = buffer_position_gather(source, dest, topo_lvl, comm, han_module);

                if(-1 != send_position) {
                    ptrdiff_t src_shift = block_size * send_position;
                    ptrdiff_t dest_shift = block_size * recv_position;
                    recv_position++;
                    ompi_datatype_sndrcv((char *)gatherv_recv_buf[last_topo_lvl] + src_shift,
                                         rcount * type_size, MPI_BYTE,
                                         (char *)alltoallv_send_buf + dest_shift,
                                         rcount * type_size, MPI_BYTE);
                }
            }
        }
        sub_comm[topo_lvl]->c_coll->coll_alltoallv(alltoallv_send_buf, alltoallv_count,
                                                   alltoallv_displ, MPI_BYTE,
                                                   alltoallv_recv_buf, alltoallv_count,
                                                   alltoallv_displ, MPI_BYTE,
                                                   sub_comm[topo_lvl],
                                                   sub_comm[topo_lvl]->c_coll->coll_alltoallv_module);

        /* Free send buffer */
        free(alltoallv_send_buf);
    }
    /* Diving scatterv loop */

    /* Scatterv buffer for each topological levels */
    void *scatterv_send_buf[GLOBAL_COMMUNICATOR-1];
    int *scatterv_send_count[GLOBAL_COMMUNICATOR-1];
    int *scatterv_send_displ[GLOBAL_COMMUNICATOR-1];
    void *scatterv_recv_buf[GLOBAL_COMMUNICATOR-1];
    int scatterv_recv_count[GLOBAL_COMMUNICATOR-1];
    void *scatterv_copy_buf[GLOBAL_COMMUNICATOR-1];

    /* Init */
    for (topo_lvl = 0; topo_lvl < (GLOBAL_COMMUNICATOR-1); topo_lvl++) {
        scatterv_send_buf[topo_lvl] = NULL;
        scatterv_send_count[topo_lvl] = NULL;
        scatterv_send_displ[topo_lvl] = NULL;
        scatterv_recv_buf[topo_lvl] = NULL;
        scatterv_recv_count[topo_lvl] = 0;
        scatterv_copy_buf[topo_lvl] = NULL;
    }

    /* Loop start at level under top_level */
    for (topo_lvl = higher_gather_lvl; topo_lvl > LEAF_LEVEL; topo_lvl--) {
        /* Process join the loop at the same level at which they left it in the gather loop */
        my_sub_rank = ompi_comm_rank(sub_comm[topo_lvl]);
        const mca_coll_han_topo_tree_node_t *topo_node = topo->my_sub_tree[topo_lvl];
        if(0 == my_sub_rank) {
            int nb_block = 0;
            scatterv_send_count[topo_lvl] = malloc(sizeof(int) * topo_node->n_sub_tree);
            scatterv_send_displ[topo_lvl] = malloc(sizeof(int) * topo_node->n_sub_tree);
            if (NULL == scatterv_send_count[topo_lvl] || NULL == scatterv_send_displ[topo_lvl]) {
                goto cleanup_and_fallback_scatter;
            }
            scatterv_send_displ[topo_lvl][0] = 0;

            /* Search size of buffer in topo struct */
            for (int sub_tree_id = 0; sub_tree_id < topo_node->n_sub_tree; sub_tree_id++) {
                scatterv_send_count[topo_lvl][sub_tree_id] = (world_size - topo_node->sub_tree[sub_tree_id].wranks_range.nb)
                                                             * topo_node->sub_tree[sub_tree_id].wranks_range.nb * rcount * type_size;
                if (sub_tree_id > 0) {
                    scatterv_send_displ[topo_lvl][sub_tree_id] = scatterv_send_count[topo_lvl][sub_tree_id-1]
                                                                 + scatterv_send_displ[topo_lvl][sub_tree_id-1];
                }
                nb_block += (world_size - topo_node->sub_tree[sub_tree_id].wranks_range.nb)
                            * topo_node->sub_tree[sub_tree_id].wranks_range.nb;
            }
            scatterv_send_buf[topo_lvl] = malloc(nb_block * rcount * type_size);
            if (NULL == scatterv_send_buf[topo_lvl]) {
                goto cleanup_and_fallback_scatter;
            }
            /* If we are at GLOBAL_COMMUNICATOR-2
             * data need to be pick in alltoallv recv buffer
             */
            if(topo_lvl == higher_gather_lvl && was_in_alltoall) {
                scatterv_copy_buf[topo_lvl] = alltoallv_recv_buf;
            } else {
                scatterv_copy_buf[topo_lvl] = scatterv_recv_buf[last_topo_lvl];
            }
            recv_position = 0;
            enum data_provider buffer_owner;

            /* data pack in scatter send buffer
             * left data in previous recv buffer
             * are also add to the send buffer
             */
            for (int dest_pos = topo_node->wranks_range.start;
                 dest_pos < topo_node->wranks_range.start
                            + topo_node->wranks_range.nb;
                 dest_pos++) {
                int dest = topo->rank_from_topo_index[dest_pos];
                for (int source_pos = 0; source_pos < world_size; source_pos++) {
                    int source = topo->rank_from_topo_index[source_pos];

                    /* Where do we find data */
                    int send_position = buffer_position_scatter(source, dest, topo_lvl, comm, han_module, &buffer_owner);
                    if (IDONTCARE == buffer_owner) {
                        continue;
                    }
                    ptrdiff_t src_shift = block_size * send_position;
                    ptrdiff_t dest_shift = block_size * recv_position;
                    recv_position++;

                    if (MYSELF == buffer_owner) {
                        /* Data left from gather recv buffer */
                        ompi_datatype_sndrcv((char *)gatherv_recv_buf[topo_lvl] + src_shift,
                                             rcount * type_size, MPI_BYTE,
                                             (char *)scatterv_send_buf[topo_lvl] + dest_shift,
                                             rcount * type_size, MPI_BYTE);
                    } else {
                        /* TODO: do not copy data from alltoallv buffer if the scatterv does not send
                         * them at this topo_lvl. Notice that will strongly complexify data search
                         * for next scatterv */
                         /* Data from the last recv buffer (scatter or alltoall) */
                        ompi_datatype_sndrcv((char *)scatterv_copy_buf[topo_lvl] + src_shift,
                                             rcount * type_size, MPI_BYTE,
                                             (char *)scatterv_send_buf[topo_lvl] + dest_shift,
                                             rcount * type_size, MPI_BYTE);
                    }
                }
            }
        }

        scatterv_recv_count[topo_lvl] = (world_size - topo_node->sub_tree[my_sub_rank].wranks_range.nb)
                                        * topo_node->sub_tree[my_sub_rank].wranks_range.nb * rcount * type_size;
        scatterv_recv_buf[topo_lvl] = malloc(scatterv_recv_count[topo_lvl]);
        if (NULL == scatterv_recv_buf[topo_lvl]) {
            goto cleanup_and_fallback_scatter;
        }
        sub_comm[topo_lvl]->c_coll->coll_scatterv(scatterv_send_buf[topo_lvl],
                                                  scatterv_send_count[topo_lvl],
                                                  scatterv_send_displ[topo_lvl],
                                                  MPI_BYTE,
                                                  scatterv_recv_buf[topo_lvl],
                                                  scatterv_recv_count[topo_lvl],
                                                  MPI_BYTE, 0,
                                                  sub_comm[topo_lvl],
                                                  sub_comm[topo_lvl]->c_coll->coll_scatterv_module);
        last_topo_lvl = topo_lvl;
    }

    /* Last scatter at leaf level with all processes*/
    my_sub_rank = ompi_comm_rank(sub_comm[LEAF_LEVEL]);
    sub_comm_size = ompi_comm_size(sub_comm[LEAF_LEVEL]);
    const mca_coll_han_topo_tree_node_t *topo_node = topo->my_sub_tree[LEAF_LEVEL];

    /* root init */
    if (0 == my_sub_rank) {
        scatterv_send_count[LEAF_LEVEL] = malloc(sub_comm_size * sizeof(int));
        scatterv_send_displ[LEAF_LEVEL] = malloc(sub_comm_size * sizeof(int));
        if (NULL == scatterv_send_count[LEAF_LEVEL] || NULL == scatterv_send_displ[LEAF_LEVEL]) {
            goto cleanup_and_fallback_scatter;
        }
        for (int leaf_rank = 0; leaf_rank < sub_comm_size; leaf_rank++) {
            scatterv_send_count[LEAF_LEVEL][leaf_rank] = (world_size - 1)
                                                         * rcount * type_size;
            scatterv_send_displ[LEAF_LEVEL][leaf_rank] = (world_size - 1)
                                                         * rcount * type_size
                                                         * leaf_rank;
        }
        scatterv_send_buf[LEAF_LEVEL] = malloc((world_size - 1)
                                               * topo_node->wranks_range.nb
                                               * rcount * type_size);
        if (NULL == scatterv_send_buf[LEAF_LEVEL]) {
            goto cleanup_and_fallback_scatter;
        }
        /* Where find last data */
        if (0 == higher_gather_lvl) {
            scatterv_copy_buf[LEAF_LEVEL] = alltoallv_recv_buf;
        } else {
            scatterv_copy_buf[LEAF_LEVEL] = scatterv_recv_buf[last_topo_lvl];
        }

        recv_position = 0;

        /* Data pack in the last scatter send buffer */
        for (int dest_pos = 0; dest_pos < world_size; dest_pos++) {
            int dest = topo->rank_from_topo_index[dest_pos];

            for (int source_pos = 0; source_pos < world_size; source_pos++) {
                int source = topo->rank_from_topo_index[source_pos];
                enum data_provider buffer_owner;

                int send_position = buffer_position_scatter(source, dest, LEAF_LEVEL, comm, han_module, &buffer_owner);
                if (IDONTCARE == buffer_owner) {
                    continue;
                }

                ptrdiff_t src_shift = block_size * send_position;
                ptrdiff_t dest_shift = block_size * recv_position;
                recv_position++;

                if (MY_PARENT == buffer_owner) {
                    ompi_datatype_sndrcv(scatterv_copy_buf[LEAF_LEVEL] + src_shift,
                                         rcount * type_size, MPI_BYTE,
                                         scatterv_send_buf[LEAF_LEVEL] + dest_shift,
                                         rcount * type_size, MPI_BYTE);
                } else {
                    ompi_datatype_sndrcv(gatherv_recv_buf[LEAF_LEVEL] + src_shift,
                                         rcount * type_size, MPI_BYTE,
                                         scatterv_send_buf[LEAF_LEVEL] + dest_shift,
                                         rcount * type_size, MPI_BYTE);
                }
            }
        }
    }

    scatterv_recv_buf[LEAF_LEVEL] = malloc((world_size - 1) * rcount * type_size);
    if (NULL == scatterv_recv_buf[LEAF_LEVEL]) {
        goto cleanup_and_fallback_scatter;
    }
    scatterv_recv_count[LEAF_LEVEL] = (world_size - 1) * rcount * type_size;
    sub_comm[LEAF_LEVEL]->c_coll->coll_scatterv(scatterv_send_buf[LEAF_LEVEL],
                                                scatterv_send_count[LEAF_LEVEL],
                                                scatterv_send_displ[LEAF_LEVEL],
                                                MPI_BYTE,
                                                scatterv_recv_buf[LEAF_LEVEL],
                                                scatterv_recv_count[LEAF_LEVEL],
                                                MPI_BYTE, 0,
                                                sub_comm[LEAF_LEVEL],
                                                sub_comm[LEAF_LEVEL]->c_coll->coll_scatterv_module);

    /* Last pack and reorder, process data from send buffer is pick and pack into the recv buffer
     * MPI_BYTE is also convert into the rdtype and data are reordered */
    ptrdiff_t rextent;
    ompi_datatype_type_extent(rdtype, &rextent);

    for (int rank = 0; rank < world_size; rank++) {
        ptrdiff_t src_shift;
        if (topo->topo_index_from_rank[rank] > topo->topo_index_from_rank[world_rank]) {
            src_shift = block_size * topo->topo_index_from_rank[rank] - block_size;
        } else {
            src_shift = block_size * topo->topo_index_from_rank[rank];
        }

        ptrdiff_t dest_shift = rcount * rextent * rank;
        if (world_rank == rank) {
            ompi_datatype_sndrcv(sbuf + scount * sextent * world_rank,
                                 scount, sdtype,
                                 rbuf+ dest_shift, rcount, rdtype);
        } else {
            ompi_datatype_sndrcv(scatterv_recv_buf[LEAF_LEVEL] + src_shift,
                                 rcount * type_size, MPI_BYTE,
                                 rbuf + dest_shift, rcount, rdtype);
        }
    }

    /* Free all temporary buffer created */

    /* Alltoall recv buffer */
    free(alltoallv_recv_buf);

    /* Gather and scatter buffer */
    for (topo_lvl = 0; topo_lvl < (GLOBAL_COMMUNICATOR-1); topo_lvl++) {
        free(gatherv_send_buf[topo_lvl]);
        free(gatherv_recv_buf[topo_lvl]);
        free(gatherv_recv_count[topo_lvl]);
        free(gatherv_recv_displ[topo_lvl]);
        free(scatterv_send_buf[topo_lvl]);
        free(scatterv_send_count[topo_lvl]);
        free(scatterv_send_displ[topo_lvl]);
        free(scatterv_recv_buf[topo_lvl]);
    }

    return OMPI_SUCCESS;

    /* If malloc fail, clean buffer and fallback to Tuned */
cleanup_and_fallback_scatter:
    for (topo_lvl = 0; topo_lvl < (GLOBAL_COMMUNICATOR-1); topo_lvl++) {
        if (NULL != scatterv_send_buf[topo_lvl]) {
            free(scatterv_send_buf[topo_lvl]);
        }
        if (NULL != scatterv_recv_buf[topo_lvl]) {
            free(scatterv_recv_buf[topo_lvl]);
        }
        if (NULL != scatterv_send_count[topo_lvl]) {
            free(scatterv_send_count[topo_lvl]);
        }
        if (NULL != scatterv_send_displ[topo_lvl]) {
            free(scatterv_send_displ[topo_lvl]);
        }
    }
cleanup_and_fallback_alltoall:
    if (NULL != alltoallv_recv_buf) {
        free(alltoallv_recv_buf);
    }
cleanup_and_fallback_gather:
    for (topo_lvl = 0; topo_lvl < (GLOBAL_COMMUNICATOR-1); topo_lvl++) {
        if(NULL != gatherv_send_buf[topo_lvl]) {
            free(gatherv_send_buf[topo_lvl]);
        }
        if(NULL != gatherv_recv_buf[topo_lvl]) {
            free(gatherv_recv_buf[topo_lvl]);
        }
        if(NULL != gatherv_recv_count[topo_lvl]) {
            free(gatherv_recv_count[topo_lvl]);
        }
        if(NULL != gatherv_recv_displ[topo_lvl]) {
            free(gatherv_recv_displ[topo_lvl]);
        }
    }
    opal_output_verbose(30, mca_coll_han_component.han_output,
                        "han failed to create buffer for alltoall GAS0.\n");
    return OMPI_ERROR;
}
