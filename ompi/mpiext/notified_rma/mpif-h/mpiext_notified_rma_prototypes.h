/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 * This file prototypes all MPI fortran functions in all four fortran
 * symbol conventions as well as all the internal real OMPI wrapper
 * functions (different from any of the four fortran symbol
 * conventions for clarity, at the cost of more typing for me...).
 * This file is included in the top-level build ONLY.
 *
 * This is needed ONLY if the lower-level prototypes_pmpi.h has not
 * already been included.
 *
 * Note about function pointers: all function pointers are prototyped
 * here as (void*) rather than including the .h file that defines the
 * proper type (e.g., "op/op.h" defines ompi_op_fortran_handler_fn_t,
 * which is the function pointer type for fortran op callback
 * functions).  This is because there is no type checking coming in
 * from fortran, so why bother?  Also, including "op/op.h" (and
 * friends) makes the all the f77 bindings files dependant on these
 * files -- any change to any one of them will cause the recompilation
 * of the entire set of f77 bindings (ugh!).
 */

#ifndef OMPI_F77_PROTOTYPES_MPI_H
#define OMPI_F77_PROTOTYPES_MPI_H

#include "ompi_config.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/attribute/attribute.h"
#include "ompi/op/op.h"
#include "ompi/request/grequest.h"
#include "ompi/mpi/fortran/base/datarep.h"
BEGIN_C_DECLS

/* These are the prototypes for the "real" back-end fortran functions. */
#define PN2(ret, mixed_name, lower_name, upper_name, args) \
    /* Prototype the actual OMPI function */               \
    OMPI_DECLSPEC ret o##lower_name##_f args;              \
    /* Prototype the 4 versions of the MPI mpif.h name */  \
    OMPI_DECLSPEC ret lower_name args;                     \
    OMPI_DECLSPEC ret lower_name##_ args;                  \
    OMPI_DECLSPEC ret lower_name##__ args;                 \
    OMPI_DECLSPEC ret upper_name args;                     \
    /* Prototype the use mpi/use mpi_f08 names  */         \
    OMPI_DECLSPEC ret mixed_name##_f08 args;               \
    OMPI_DECLSPEC ret mixed_name##_f args;                 \
    /* Prototype the actual POMPI function */              \
    OMPI_DECLSPEC ret po##lower_name##_f args;             \
    /* Prototype the 4 versions of the PMPI mpif.h name */ \
    OMPI_DECLSPEC ret p##lower_name args;                  \
    OMPI_DECLSPEC ret p##lower_name##_ args;               \
    OMPI_DECLSPEC ret p##lower_name##__ args;              \
    OMPI_DECLSPEC ret P##upper_name args;                  \
    /* Prototype the use mpi/use mpi_f08 PMPI names  */    \
    OMPI_DECLSPEC ret P##mixed_name##_f08 args;            \
    OMPI_DECLSPEC ret P##mixed_name##_f args

PN2(void, MPIX_Get_notify, mpix_get_notify, MPIX_GET_NOTIFY, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr));
PN2(void, MPIX_Put_notify, mpix_put_notify, MPIX_PUT_NOTIFY, (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype, MPI_Fint *target_rank, MPI_Aint *target_disp, MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr));
PN2(void, MPIX_Win_allocate_notify, mpix_win_allocate_notify, MPIX_WIN_ALLOCATE_NOTIFY, (MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, char *baseptr, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPIX_Win_create_notify, mpix_win_create_notify, MPIX_WIN_CREATE_NOTIFY, (char *base, MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info, MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr));
PN2(void, MPIX_Win_test_notify, mpix_win_test_notify, MPIX_WIN_TEST_NOTIFY, (MPI_Fint *win, MPI_Fint *notification_id, ompi_fortran_logical_t *flag, MPI_Fint *ierr));
PN2(void, MPIX_Win_wait_notify, mpix_win_wait_notify, MPIX_WIN_WAIT_NOTIFY, (MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr));
END_C_DECLS

#endif
