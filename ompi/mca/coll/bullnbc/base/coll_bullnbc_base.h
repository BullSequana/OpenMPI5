/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.\n"
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef __MCA_COLL_BULLNBC_BASE_H__
#define __MCA_COLL_BULLNBC_BASE_H__


/* base allgatherv algorithms */


/* base barrier algorithms */

int ompi_coll_bullnbc_base_ibarrier_intra_doublering(struct ompi_communicator_t *comm,
                                                     ompi_request_t ** request,
                                                     struct mca_coll_base_module_2_4_0_t *module,
                                                     bool persistent);

int ompi_coll_bullnbc_base_barrier_intra_recursivedoubling(struct ompi_communicator_t *comm,
                                                           ompi_request_t ** request,
                                                           struct mca_coll_base_module_2_4_0_t *module,
                                                           bool persistent);

int ompi_coll_bullnbc_base_ibarrier_intra_two_procs(struct ompi_communicator_t *comm,
                                                    ompi_request_t ** request,
                                                    struct mca_coll_base_module_2_4_0_t *module,
                                                    bool persistent);

int ompi_coll_bullnbc_base_ibarrier_intra_recursivedoubling(struct ompi_communicator_t *comm,
                                                            ompi_request_t ** request,
                                                            struct mca_coll_base_module_2_4_0_t *module,
                                                            bool persistent);

int ompi_coll_bullnbc_base_ibarrier_intra_bruck(struct ompi_communicator_t *comm,
                                                ompi_request_t ** request,
                                                struct mca_coll_base_module_2_4_0_t *module,
                                                bool persistent);

int ompi_coll_bullnbc_base_ibarrier_intra_basic_linear(struct ompi_communicator_t *comm,
                                                       ompi_request_t ** request,
                                                       struct mca_coll_base_module_2_4_0_t *module,
                                                       bool persistent);

int ompi_coll_bullnbc_base_ibarrier_intra_tree(struct ompi_communicator_t *comm,
                                               ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module,
                                               bool persistent);
#endif /* __MCA_COLL_BULLNBC_BASE_H__ */
