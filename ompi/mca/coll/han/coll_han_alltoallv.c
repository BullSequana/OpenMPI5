/*
 * Copyright (c) 2023-2024 BULL S.A.S. All rights reserved.
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

/* 2 levels alltoallv algorithm based on grid alltoall algorithm.
 * First, data is sent packed through up_comm according to target ranks nodes (alltoallv).
 * Then, remote ranks dispatch data locally to their neighbors on low_comm (alltoallw).
 * This is performed in parallel across all up_comms and low_comms.
 * Only need a balanced topology.
 * Alltoallw at each topo level starting at INTER NODE level with hindexed datatypes,
 * need an alltoall before first alltoallw to share count with other rank.
 *
 * For example, let's consider a topology as follows (3 nodes, 2 ranks each):
 * [[0 1] [2 5] [3 4]]
 * If "XY" is the data from X to Y and |XY| its length
 * rank 4 will send |40||41| and |42||45| to 1 and 5 in a first alltoall and it will receive |13||14| and |53||54|.
 * Once the lengths are known, the data may be properly received in an alltoallw on up_comm.
 * During a second alltoallw (low_comm), rank 4 will send "13" and "53" to 3, and receive "04" and "24". (No reordering, just picking)
 * Correct reordering is made on this last reception, using an hindexed datatype.
 * For each alltoallw data selection uses hindexed datatypes.
 */
int mca_coll_han_alltoallv_grid(const void *sbuf,
                                const int *scount,
                                const int *sdispl,
                                struct ompi_datatype_t *sdtype,
                                void *rbuf,
                                const int *rcount,
                                const int *rdispl,
                                struct ompi_datatype_t *rdtype,
                                struct ompi_communicator_t *comm,
                                mca_coll_base_module_t *module)
{
    /* Check if we can run alltoallv grid */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle alltoallv with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }
    int err;
    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoallv with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }

    /* We need a balanced topology to run this algorithm */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoallv with this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, bcast);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }
    /* Take information about topology and datatype */
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);
    int w_size = ompi_comm_size(comm);
    size_t type_size;
    ptrdiff_t sdtype_extent;
    ptrdiff_t rdtype_extent;

    /* MPI IN PLACE support */
    if (sbuf == MPI_IN_PLACE) {
        scount = rcount;
        sdispl = rdispl;
        sdtype = rdtype;
        sbuf = rbuf;
    }
    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);
    ompi_datatype_type_size(sdtype, &type_size);
    /* On each up_comm, ranks receive data for their neighbor ranks.
     * This inter-node alltoall shares byte counts for each local peers to intermediate ranks.  */
    size_t *up_buf_length;
    /* Store datatype size in bytes for each blocks */
    up_buf_length = malloc(sizeof(size_t) * w_size);
    /* Avoid reordering with topological information */
    const int *rank_by_locality = han_module->topo_tree->rank_from_topo_index;
    for (int topo_index = 0; topo_index < w_size; topo_index++) {
        up_buf_length[topo_index] = (size_t)scount[rank_by_locality[topo_index]] * type_size;
    }
    /* up_comm ranks already follow topological order, no further counts reordering is needed */
    up_comm->c_coll->coll_alltoall(MPI_IN_PLACE,
                                   low_size,
                                   MPI_UNSIGNED_LONG,
                                   up_buf_length,
                                   low_size,
                                   MPI_UNSIGNED_LONG,
                                   up_comm,
                                   up_comm->c_coll->coll_alltoall_module);

    /* First alltoallw with sdtype -> MPI_BYTE*/
    const void *up_sbuf = sbuf;
    void *up_rbuf = NULL;
    /* This will contain all int information for up alltoallw
     * This is used to prevent too many malloc/free*/
    int *up_alltoallw_int;
    up_alltoallw_int = malloc(4 * up_size * sizeof(int));
    /* Start position of data */
    int *up_scount = up_alltoallw_int;
    int *up_rcount = up_alltoallw_int + up_size;
    int *up_sdispl = up_alltoallw_int + 2 * up_size;
    int *up_rdispl = up_alltoallw_int + 3 * up_size;
    /* This will contain all the datatype's information for up alltoallw */
    struct ompi_datatype_t **up_alltoallw_dtype;
    up_alltoallw_dtype = malloc(2 * up_size * sizeof(struct ompi_datatype_t *));
    /* Start position of data */
    struct ompi_datatype_t **up_sdtype = up_alltoallw_dtype;
    struct ompi_datatype_t **up_rdtype = up_alltoallw_dtype + up_size;
    /* Temporary buffer information */
    size_t up_recv_packed_size;
    size_t up_recv_buffer_size = 0;
    /* Use Hindexed datatype to select data we need to send and keep in sendbuff*/
    int *up_hindexed_lengths;
    up_hindexed_lengths = malloc(sizeof(int) * low_size);
    ptrdiff_t *up_hindexed_displ;
    up_hindexed_displ = malloc(sizeof(ptrdiff_t) * low_size);
    for (int up_rank = 0; up_rank < up_size; up_rank++) {
        up_recv_packed_size = 0;
        up_scount[up_rank] = 1;
        up_sdispl[up_rank] = 0;
        for (int low_rank = 0; low_rank < low_size; low_rank++) {
            up_hindexed_lengths[low_rank] = scount[rank_by_locality[up_rank * low_size + low_rank]];
            up_hindexed_displ[low_rank] = (ptrdiff_t)sdispl[rank_by_locality[up_rank * low_size + low_rank]] * sdtype_extent;
            up_recv_packed_size += up_buf_length[up_rank * low_size + low_rank];
        }
        ompi_datatype_create_hindexed(low_size, up_hindexed_lengths, up_hindexed_displ, sdtype, &up_sdtype[up_rank]);
        ompi_datatype_commit(&up_sdtype[up_rank]);
        up_rdtype[up_rank] = MPI_BYTE;
        up_rcount[up_rank] = (int)up_recv_packed_size;
        if (up_rank == 0) {
            up_rdispl[up_rank] = 0;
        } else {
            up_rdispl[up_rank] = up_rdispl[up_rank - 1] + up_rcount[up_rank - 1];
        }
        /* Compute size of temporary recvbuf */
        up_recv_buffer_size += up_recv_packed_size;
    }
    free(up_hindexed_lengths);
    free(up_hindexed_displ);
    /* We can't use Rbuf because it can be too small or already used as sbuf (MPI_IN_PLACE) */
    if (up_recv_buffer_size > 0) {
        up_rbuf = malloc(up_recv_buffer_size);
    }

    up_comm->c_coll->coll_alltoallw(up_sbuf,
                                    up_scount,
                                    up_sdispl,
                                    up_sdtype,
                                    up_rbuf,
                                    up_rcount,
                                    up_rdispl,
                                    up_rdtype,
                                    up_comm,
                                    up_comm->c_coll->coll_alltoallw_module);

    /* Sdtype can be free here*/
    for(int up_rank = 0; up_rank < up_size; up_rank++) {
        ompi_datatype_destroy(&up_sdtype[up_rank]);
    }
    free(up_alltoallw_dtype);

    /* Leaf level alltoallw MPI_BYTE -> rdtype*/
    const void *low_sbuf = up_rbuf;
    void *low_rbuf = rbuf;
    /* This will contain all int information for low alltoallw*/
    int *low_alltoallw_int;
    low_alltoallw_int = malloc(2 * low_size * sizeof(int));
    /* Start position of data */
    int *low_count = low_alltoallw_int;
    int *low_displ = low_alltoallw_int + low_size;
    /* This will contain all the datatype's information for low alltoallw */
    struct ompi_datatype_t **low_alltoallw_dtype;
    low_alltoallw_dtype = malloc(2 * low_size * sizeof(struct ompi_datatype_t *));
    /* Start position of data */
    struct ompi_datatype_t **low_sdtype = low_alltoallw_dtype;
    struct ompi_datatype_t **low_rdtype = low_alltoallw_dtype + low_size;

    int *low_sbuf_block_displ;
    low_sbuf_block_displ = malloc (sizeof(int) * up_size);

    /* Datatype parameters */
    int *low_hindexed_lengths;
    low_hindexed_lengths = malloc(sizeof(int) * up_size * 2);
    ptrdiff_t *low_hindexed_displ;
    low_hindexed_displ = malloc(sizeof(ptrdiff_t) * up_size * 2);
    /* Start position */
    int *length_send = low_hindexed_lengths;
    int *length_recv = low_hindexed_lengths + up_size;
    ptrdiff_t *displ_send = low_hindexed_displ;
    ptrdiff_t *displ_recv = low_hindexed_displ + up_size;

    for (int low_rank = 0; low_rank < low_size; low_rank++) {
        low_count[low_rank] = 1;
        low_displ[low_rank] = 0;
        for (int up_rank = 0; up_rank < up_size; up_rank++) {
            length_send[up_rank] = (int)up_buf_length[up_rank * low_size + low_rank];
            length_recv[up_rank] = rcount[rank_by_locality[up_rank * low_size + low_rank]];
            if (low_rank == 0) {
                displ_send[up_rank] = (ptrdiff_t)up_rdispl[up_rank];
                low_sbuf_block_displ[up_rank] = length_send[up_rank];
            } else {
                displ_send[up_rank] = (ptrdiff_t)up_rdispl[up_rank] + (ptrdiff_t)low_sbuf_block_displ[up_rank];
                low_sbuf_block_displ[up_rank] += length_send[up_rank];
            }
            displ_recv[up_rank] = (ptrdiff_t)rdispl[rank_by_locality[up_rank * low_size + low_rank]] * rdtype_extent;
        }
        ompi_datatype_create_hindexed(up_size, length_send, displ_send, MPI_BYTE, &low_sdtype[low_rank]);
        ompi_datatype_commit(&low_sdtype[low_rank]);
        ompi_datatype_create_hindexed(up_size, length_recv, displ_recv, rdtype, &low_rdtype[low_rank]);
        ompi_datatype_commit(&low_rdtype[low_rank]);
    }

    free(low_hindexed_lengths);
    free(low_hindexed_displ);
    free(up_buf_length);

    low_comm->c_coll->coll_alltoallw(low_sbuf,
                                     low_count,
                                     low_displ,
                                     low_sdtype,
                                     low_rbuf,
                                     low_count,
                                     low_displ,
                                     low_rdtype,
                                     low_comm,
                                     low_comm->c_coll->coll_alltoallw_module);

    for(int low_rank = 0; low_rank < low_size; low_rank++) {
        ompi_datatype_destroy(&low_sdtype[low_rank]);
        ompi_datatype_destroy(&low_rdtype[low_rank]);
    }
    free(up_alltoallw_int);
    free(low_alltoallw_int);
    free(low_alltoallw_dtype);
    free(low_sbuf_block_displ);
    free(up_rbuf);
    return OMPI_SUCCESS;
}

/* Grid algorithm but all blocks sent are cut into segment with MCA parameter 
 * (OMPI_MCA_coll_han_alltoallv_pipeline_segment_count). */
int mca_coll_han_alltoallv_grid_pipeline(const void *sbuf, // NOSONAR complexity
                                         const int *scount,
                                         const int *sdispl,
                                         struct ompi_datatype_t *sdtype,
                                         void *rbuf,
                                         const int *rcount,
                                         const int *rdispl,
                                         struct ompi_datatype_t *rdtype,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    /* Check if we can run alltoallv grid */
    mca_coll_han_module_t *han_module = (mca_coll_han_module_t *)module;
    if (!mca_coll_han_has_2_levels(han_module)) {
        opal_output_verbose(30, mca_coll_han_component.han_output,
                             "han cannot handle alltoallv with this communicator (not 2 levels). Fall back on another component\n");
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoallv);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }

    int err;
    /* Create the subcommunicators */
    err = mca_coll_han_comm_create_multi_level(comm, han_module);
    if( OMPI_SUCCESS != err ) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle any collective with this communicator. Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVES(han_module, comm);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }

    /* We need a balanced topology to run this algorithm */
    if (han_module->are_ppn_imbalanced) {
        OPAL_OUTPUT_VERBOSE((30, mca_coll_han_component.han_output,
                             "han cannot handle alltoallv with this communicator (imbalance). "
                             "Fall back on another component\n"));
        /* Put back the fallback collective support and call it once. All
         * future calls will then be automatically redirected.
         */
        HAN_LOAD_FALLBACK_COLLECTIVE(han_module, comm, alltoallv);
        return comm->c_coll->coll_alltoallv(sbuf, scount, sdispl, sdtype,
                                            rbuf, rcount, rdispl, rdtype,
                                            comm, comm->c_coll->coll_alltoallv_module);
    }

    /* Take information about topology and datatype */
    ompi_communicator_t *low_comm = han_module->sub_comm[LEAF_LEVEL];
    ompi_communicator_t *up_comm = han_module->sub_comm[INTER_NODE];
    int low_size = ompi_comm_size(low_comm);
    int up_size = ompi_comm_size(up_comm);
    int w_size = ompi_comm_size(comm);
    size_t sdtype_size;
    size_t rdtype_size;
    ptrdiff_t sdtype_extent;
    ptrdiff_t rdtype_extent;

    /* MPI IN PLACE support */
    if (sbuf == MPI_IN_PLACE) {
        scount = rcount;
        sdispl = rdispl;
        sdtype = rdtype;
        sbuf = rbuf;
    }
    ompi_datatype_type_extent(sdtype, &sdtype_extent);
    ompi_datatype_type_extent(rdtype, &rdtype_extent);
    ompi_datatype_type_size(sdtype, &sdtype_size);
    ompi_datatype_type_size(rdtype, &rdtype_size);

    /* Pipeline information */
    int nb_segment;
    nb_segment = mca_coll_han_component.alltoallv_pipeline_segment_count;

    /* On each up_comm and low_comm, ranks receive data for their neighbor ranks.
     * This inter-node alltoall shares byte counts for each local peers to intermediate ranks.  */
    size_t *up_buf_length;
    size_t *low_buf_length;
    /* Store datatype size in bytes for each blocks */
    up_buf_length = malloc(sizeof(size_t) * w_size * 2);
    low_buf_length = malloc(sizeof(size_t) * w_size * 2);
    /* Avoid reordering with topological information */
    const int *rank_by_locality = han_module->topo_tree->rank_from_topo_index;
    for (int topo_index = 0; topo_index < w_size; topo_index++) {
        up_buf_length[topo_index * 2] = sdtype_size;
        up_buf_length[topo_index * 2 + 1] = (size_t)scount[rank_by_locality[topo_index]];
        low_buf_length[topo_index * 2] = rdtype_size;
        low_buf_length[topo_index * 2 + 1] = (size_t)rcount[rank_by_locality[topo_index]];
    }

    /* We need to exchange data according to the up_comm, so we need vector */
    ompi_request_t *ialltoall_req;
    struct ompi_datatype_t *vector_not_resized;
    struct ompi_datatype_t *vector_resized;
    ompi_datatype_create_vector(up_size, 2, low_size * 2, MPI_UINT64_T, &vector_not_resized);
    ompi_datatype_commit(&vector_not_resized);
    ptrdiff_t ul_extent;
    ompi_datatype_type_extent(MPI_UINT64_T, &ul_extent);
    ompi_datatype_create_resized(vector_not_resized, 0, 2 * ul_extent, &vector_resized);
    ompi_datatype_commit(&vector_resized);

    low_comm->c_coll->coll_ialltoall(MPI_IN_PLACE,
                                     1,
                                     vector_resized,
                                     low_buf_length,
                                     1,
                                     vector_resized,
                                     low_comm,
                                     &ialltoall_req,
                                     low_comm->c_coll->coll_ialltoall_module);
    
    /* up_comm ranks already follow topological order, no further counts reordering is needed */
    up_comm->c_coll->coll_alltoall(MPI_IN_PLACE,
                                   2 * low_size,
                                   MPI_UNSIGNED_LONG,
                                   up_buf_length,
                                   2 * low_size,
                                   MPI_UNSIGNED_LONG,
                                   up_comm,
                                   up_comm->c_coll->coll_alltoall_module);

    /*Prepare tmp buff and its future displacement start, we need to compute its total size before the start of pipeline */
    /* There are 2 displacement in this array : 2   is the current displ for up_comm recv part
     *                                          2+1 is the current displ for low_comm send part */
    void *up_rbuf = NULL;
    size_t tmp_buf_size = 0;
    ptrdiff_t *tmp_buf_displ;
    size_t *recv_displ_offset;
    tmp_buf_displ = malloc(sizeof(ptrdiff_t) * w_size * 2);
    recv_displ_offset = malloc(sizeof(size_t) * w_size);
    for (int peers = 0; peers < w_size; peers++) {
        tmp_buf_displ[peers * 2] = tmp_buf_size;
        tmp_buf_displ[peers * 2 + 1] = tmp_buf_size;
        tmp_buf_size += up_buf_length[peers * 2] * up_buf_length[peers * 2 + 1];
        recv_displ_offset[peers] = 0;
    }
    if (tmp_buf_size != 0) {
        up_rbuf = malloc(tmp_buf_size);
    }
    /* Pipeline start */
    ompi_request_t **req;
    req = (ompi_request_t**) malloc(nb_segment * sizeof(struct ompi_request_t *));

    for (int iter = 0; iter < nb_segment; iter++)  {
         /* First alltoallw with sdtype -> MPI_BYTE*/
         const void *up_sbuf = sbuf;

         /* This will contain all int information for up alltoallw
          * This is used to prevent too many malloc/free*/
         int *up_alltoallw_int;
         up_alltoallw_int = malloc(4 * up_size * sizeof(int));

         /* Start position of data */
         int *up_scount = up_alltoallw_int;
         int *up_rcount = up_alltoallw_int + up_size;
         int *up_sdispl = up_alltoallw_int + 2 * up_size;
         int *up_rdispl = up_alltoallw_int + 3 * up_size;

         /* This will contain all the datatype's information for up alltoallw */
         struct ompi_datatype_t **up_alltoallw_dtype;
         up_alltoallw_dtype = malloc(2 * up_size * sizeof(struct ompi_datatype_t *));

         /* Start position of data */
         struct ompi_datatype_t **up_sdtype = up_alltoallw_dtype;
         struct ompi_datatype_t **up_rdtype = up_alltoallw_dtype + up_size;

         /* Use Hindexed datatype to select data we need to send and keep in sendbuff*/
         int *up_hindexed_lengths;
         up_hindexed_lengths = malloc(sizeof(int) * low_size * 2);
         ptrdiff_t *up_hindexed_displ;
         up_hindexed_displ = malloc(sizeof(ptrdiff_t) * low_size * 2);

         /* Start position */
         int *length_send = up_hindexed_lengths;
         int *length_recv = up_hindexed_lengths + low_size;
         ptrdiff_t *displ_send = up_hindexed_displ;
         ptrdiff_t *displ_recv = up_hindexed_displ + low_size;

         for (int up_rank = 0; up_rank < up_size; up_rank++) {
             up_scount[up_rank] = 1;
             up_sdispl[up_rank] = 0;
             for (int low_rank = 0; low_rank < low_size; low_rank++) {
                 int real_rank = rank_by_locality[up_rank * low_size + low_rank];
                 int datatype_size = (int)up_buf_length[(up_rank * low_size + low_rank) * 2];
                 int pipeline_scount = scount[real_rank] / nb_segment;
                 int pipeline_rcount = (int)up_buf_length[(up_rank * low_size + low_rank) * 2 + 1] / nb_segment;
                 int remain_rcount = up_buf_length[(up_rank * low_size + low_rank) * 2 + 1] % nb_segment;
                 int remain_scount = scount[real_rank] % nb_segment;

                 /* Always check if segmentation is perfect, if it's not the case,
                  * we distribute remain in each block send at the start of the pipeline*/
                 if (remain_scount <= iter) {
                    length_send[low_rank] = pipeline_scount;
                     displ_send[low_rank] = (ptrdiff_t)sdispl[real_rank] * sdtype_extent + sdtype_extent * (pipeline_scount * iter +  remain_scount);
                 } else {
                    length_send[low_rank] = pipeline_scount + 1;
                     displ_send[low_rank] = (ptrdiff_t)sdispl[real_rank] * sdtype_extent + sdtype_extent * (pipeline_scount * iter + iter);
                 }
                 if (remain_rcount <= iter) {
                    length_recv[low_rank] = pipeline_rcount * datatype_size;
                 } else {
                    length_recv[low_rank] = (pipeline_rcount + 1) * datatype_size;
                 }
                 displ_recv[low_rank] = tmp_buf_displ[(up_rank * low_size + low_rank) * 2];//NOSONAR (False out of bound here)
                 tmp_buf_displ[(up_rank * low_size + low_rank) * 2] += (ptrdiff_t)length_recv[low_rank]; // NOSONAR left expr uninitialized
             }
             ompi_datatype_create_hindexed(low_size, length_send, displ_send, sdtype, &up_sdtype[up_rank]);
             ompi_datatype_commit(&up_sdtype[up_rank]);
             ompi_datatype_create_hindexed(low_size, length_recv, displ_recv, MPI_BYTE, &up_rdtype[up_rank]);
             ompi_datatype_commit(&up_rdtype[up_rank]);
             up_rcount[up_rank] = 1;
             up_rdispl[up_rank] = 0;
         }
         free(up_hindexed_lengths);
         free(up_hindexed_displ);

         up_comm->c_coll->coll_alltoallw(up_sbuf,
                                         up_scount,
                                         up_sdispl,
                                         up_sdtype,
                                         up_rbuf,
                                         up_rcount,
                                         up_rdispl,
                                         up_rdtype,
                                         up_comm,
                                         up_comm->c_coll->coll_alltoallw_module);
         /* Up information can be free here*/
         for(int up_rank = 0; up_rank < up_size; up_rank++) {
             ompi_datatype_destroy(&up_sdtype[up_rank]);
             ompi_datatype_destroy(&up_rdtype[up_rank]);
         }
         free(up_alltoallw_dtype);
         free(up_alltoallw_int);

         /* We need recv information share by ialltoall now */
         if (iter == 0) {
            ompi_request_wait(&ialltoall_req, MPI_STATUS_IGNORE);
            ompi_datatype_destroy(&vector_not_resized);
            ompi_datatype_destroy(&vector_resized);
         }

         /* Leaf level alltoallw MPI_BYTE -> rdtype*/
         const void *low_sbuf = up_rbuf;
         void *low_rbuf = rbuf;

         /* This will contain all int information for low alltoallw*/
         int *low_alltoallw_int;
         low_alltoallw_int = malloc(2 * low_size * sizeof(int));

         /* Start position of data */
         int *low_count = low_alltoallw_int;
         int *low_displ = low_alltoallw_int + low_size;

         /* This will contain all the datatype's information for low alltoallw */
         struct ompi_datatype_t **low_alltoallw_dtype;
         low_alltoallw_dtype = malloc(2 * low_size * sizeof(struct ompi_datatype_t *));

         /* Start position of data */
         struct ompi_datatype_t **low_sdtype = low_alltoallw_dtype;
         struct ompi_datatype_t **low_rdtype = low_alltoallw_dtype + low_size;

         /* Datatype parameters */
         int *low_hindexed_lengths;
         low_hindexed_lengths = malloc(sizeof(int) * up_size * 2);
         ptrdiff_t *low_hindexed_displ;
         low_hindexed_displ = malloc(sizeof(ptrdiff_t) * up_size * 2);

         /* Start position */
         length_send = low_hindexed_lengths;
         length_recv = low_hindexed_lengths + up_size;
         displ_send = low_hindexed_displ;
         displ_recv = low_hindexed_displ + up_size;

         for (int low_rank = 0; low_rank < low_size; low_rank++) {
             low_count[low_rank] = 1;
             low_displ[low_rank] = 0;
             for (int up_rank = 0; up_rank < up_size; up_rank++) {
                 int real_rank = rank_by_locality[up_rank * low_size + low_rank];
                 int pipeline_rcount = (int)low_buf_length[(up_rank * low_size + low_rank) * 2 + 1] / nb_segment;
                 int datatype_size = (int)low_buf_length[(up_rank * low_size + low_rank) * 2];
                 /* Always check if segmentation is perfect, if it's not the case,
                  * we distribute remain in each block send at the end of the pipeline*/
                 if(nb_segment - (low_buf_length[(up_rank * low_size + low_rank) * 2 + 1] % nb_segment) > iter) {
                     length_send[up_rank] = pipeline_rcount * datatype_size;
                 } else {
                     length_send[up_rank] = pipeline_rcount * datatype_size + datatype_size;
                 }
                 displ_send[up_rank] = tmp_buf_displ[(up_rank * low_size + low_rank) * 2 + 1];
                 tmp_buf_displ[(up_rank * low_size + low_rank) * 2 + 1] += (ptrdiff_t)length_send[up_rank];
                 if (nb_segment - rcount[real_rank] % nb_segment  > iter) { 
                     length_recv[up_rank] = rcount[real_rank] / nb_segment;
                 } else {
                     length_recv[up_rank] = rcount[real_rank] / nb_segment + 1;
                 }
                 displ_recv[up_rank] = (ptrdiff_t)rdispl[real_rank] * rdtype_extent + (ptrdiff_t)recv_displ_offset[up_rank * low_size + low_rank];
                 recv_displ_offset[up_rank * low_size + low_rank] += (size_t)length_recv[up_rank] * rdtype_extent;
             }
             ompi_datatype_create_hindexed(up_size, length_send, displ_send, MPI_BYTE, &low_sdtype[low_rank]);
             ompi_datatype_commit(&low_sdtype[low_rank]);
             ompi_datatype_create_hindexed(up_size, length_recv, displ_recv, rdtype, &low_rdtype[low_rank]);
             ompi_datatype_commit(&low_rdtype[low_rank]);
         }

         free(low_hindexed_lengths);
         free(low_hindexed_displ);

         low_comm->c_coll->coll_ialltoallw(low_sbuf,
                                           low_count,
                                           low_displ,
                                           low_sdtype,
                                           low_rbuf,
                                           low_count,
                                           low_displ,
                                           low_rdtype,
                                           low_comm,
                                           &req[iter],
                                           low_comm->c_coll->coll_ialltoallw_module);

         for(int low_rank = 0; low_rank < low_size; low_rank++) {
             ompi_datatype_destroy(&low_sdtype[low_rank]);
             ompi_datatype_destroy(&low_rdtype[low_rank]);
         }
         free(low_alltoallw_int);
         free(low_alltoallw_dtype);
    }
    free(up_buf_length);
    free(low_buf_length);
    for (int segment_sent = 0; segment_sent < nb_segment; segment_sent++){
        ompi_request_wait(&req[segment_sent], MPI_STATUS_IGNORE);
    }
    free(up_rbuf);
    free(tmp_buf_displ);
    free(recv_displ_offset);
    free(req);
    return OMPI_SUCCESS;
}
