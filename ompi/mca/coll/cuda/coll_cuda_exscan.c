/*
 * Copyright (c) 2014-2017 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2014-2015 NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2022      Amazon.com, Inc. or its affiliates.  All Rights reserved.
 * Copyright (c) 2024      BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_cuda.h"

#include <stdio.h>

#include "ompi/op/op.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/opal_touch_pages.h"

int mca_coll_cuda_exscan(const void *sbuf, void *rbuf, int count,
                         struct ompi_datatype_t *dtype,
                         struct ompi_op_t *op,
                         struct ompi_communicator_t *comm,
                         mca_coll_base_module_t *module)
{
    mca_coll_cuda_module_t *s = (mca_coll_cuda_module_t*) module;
    ptrdiff_t gap;
    char *rbuf1 = NULL, *sbuf1 = NULL, *rbuf2 = NULL;
    size_t bufsize;
    int rc;

    bufsize = opal_datatype_span(&dtype->super, count, &gap);
    rc = mca_coll_cuda_check_buf((void *)sbuf);
    if (rc < 0) {
        return rc;
    }

    if ((MPI_IN_PLACE != sbuf) && rc > 0) {
        sbuf1 = (char*)malloc(bufsize);
        if (NULL == sbuf1) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        if (mca_coll_cuda_component.enable_tmp_buf_touch_pages) {
            opal_touch_pages_write(sbuf1, bufsize);
        }

        if (ompi_datatype_is_contiguous_memory_layout(dtype, count)) {
            mca_coll_cuda_memcpy(sbuf1, sbuf, bufsize);
        } else {
            ompi_datatype_copy_content_same_ddt(dtype, count, sbuf1 - gap, (char *) sbuf);
        }
        sbuf = sbuf1 - gap;
    }
    rc = mca_coll_cuda_check_buf(rbuf);
    if (rc < 0) {
        return rc;
    }
    if (rc > 0) {
        rbuf1 = (char*)malloc(bufsize);
        if (NULL == rbuf1) {
            if (NULL != sbuf1) free(sbuf1);
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        if (mca_coll_cuda_component.enable_tmp_buf_touch_pages) {
            opal_touch_pages_write(rbuf1, bufsize);
        }

        if (ompi_datatype_is_contiguous_memory_layout(dtype, count)) {
            mca_coll_cuda_memcpy(rbuf1, rbuf, bufsize);
        } else {
            ompi_datatype_copy_content_same_ddt(dtype, count, rbuf1 - gap, (char *) rbuf);
        }
        rbuf2 = rbuf; /* save away original buffer */
        rbuf = rbuf1 - gap;
    }

    rc = s->c_coll.coll_exscan(sbuf, rbuf, count, dtype, op, comm,
                               s->c_coll.coll_exscan_module);
    if (NULL != sbuf1) {
        free(sbuf1);
    }
    if (NULL != rbuf1) {
        rbuf = rbuf2;
        if (ompi_datatype_is_contiguous_memory_layout(dtype, count)) {
            mca_coll_cuda_memcpy(rbuf, rbuf1, bufsize);
        } else {
            ompi_datatype_copy_content_same_ddt(dtype, count, rbuf, rbuf1 - gap);
        }
        free(rbuf1);
    }
    return rc;
}
