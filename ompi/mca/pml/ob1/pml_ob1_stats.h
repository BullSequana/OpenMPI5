/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2023-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef __PML_OB1_STATS__
#define __PML_OB1_STATS__

#include "pml_ob1.h"
#include "pml_ob1_comm.h"
#include "pml_ob1_recvreq.h"

void mca_pml_ob1_compute_depth(mca_pml_ob1_comm_t *comm, mca_pml_ob1_comm_proc_t *cur_proc,
                               mca_pml_ob1_recv_request_t *match);
void pml_ob1_stats_register_parameters(void);
#endif /* #define __PML_OB1_STATS__ */
