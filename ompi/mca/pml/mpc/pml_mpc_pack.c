/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "pml_mpc_pack.h"

int
mca_pml_mpc_pack(void* target_buf,
                 const void* origin_buf,
                 size_t count,
                 ompi_datatype_t* datatype)
{
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    opal_convertor_t convertor;
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);
    int ret = opal_convertor_copy_and_prepare_for_send(ompi_mpi_local_convertor,
                                                       &datatype->super, count, 
                                                       origin_buf, 0, &convertor);
    if (OPAL_SUCCESS != ret) {
        pml_mpc_error("opal_convertor_copy_and_prepared_for_send() failed");
        OBJ_DESTRUCT(&convertor);
        return OMPI_ERROR;
    }
    struct iovec iov;
    iov.iov_len = datatype_size * count;
    iov.iov_base = target_buf;
    uint32_t iov_count = 1;
    size_t max_data = 0;
    /* Pack data from converter to iov */
    ret = opal_convertor_pack(&convertor, &iov, &iov_count, &max_data);
    if (1 != ret) { // failure of pack is -1, incomplete pack is 0!
        pml_mpc_error("opal_convertor_pack failed");
        OBJ_DESTRUCT(&convertor);
        return OMPI_ERROR;
    }
    OBJ_DESTRUCT(&convertor);
    return OMPI_SUCCESS;
}

int
mca_pml_mpc_unpack(void* buf,
                   void* tmp_buf,
                   int count,
                   ompi_datatype_t* datatype)
{
    size_t datatype_size;
    ompi_datatype_type_size(datatype, &datatype_size);
    opal_convertor_t convertor;
    OBJ_CONSTRUCT(&convertor, opal_convertor_t);
    int ret = opal_convertor_copy_and_prepare_for_recv(ompi_mpi_local_convertor,
                                                       &datatype->super, count,
                                                       buf, 0, &convertor);
    if (OPAL_SUCCESS != ret) {
        pml_mpc_error("opal_convertor_copy_and_prepared_for_recv() failed");
        OBJ_DESTRUCT(&convertor);
        return OMPI_ERROR;
    }
    struct iovec iov;
    iov.iov_len = datatype_size * count;
    iov.iov_base = tmp_buf;
    uint32_t iov_count = 1;
    size_t max_data = 0;
    /* Unpack data to convertor from iov */
    ret = opal_convertor_unpack(&convertor, &iov, &iov_count, &max_data);
    if (-1 == ret) { // failure of unpack is -1 !
        pml_mpc_error("opal_convertor_unpack failed");
        OBJ_DESTRUCT(&convertor);
        return OMPI_ERROR;
    }
    OBJ_DESTRUCT(&convertor);
    return OMPI_SUCCESS;
}
