/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef PML_MPC_PACK_H
#define PML_MPC_PACK_H

#include "pml_mpc.h"

int
mca_pml_mpc_pack(void* target_buf,
                 const void* origin_buf,
                 size_t count,
                 ompi_datatype_t* datatype);

int
mca_pml_mpc_unpack(void* buf,
                   void* tmp_buf,
                   int count,
                   ompi_datatype_t* datatype);
#endif
