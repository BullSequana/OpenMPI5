/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/mpi/fortran/base/constants.h"

#include "ompi/mpiext/notified_rma/c/mpiext_notified_rma_c.h"

#if OMPI_BUILD_MPI_PROFILING
#    if OPAL_HAVE_WEAK_SYMBOLS
#        pragma weak PMPIX_GET_NOTIFY = ompix_get_notify_f
#        pragma weak pmpix_get_notify = ompix_get_notify_f
#        pragma weak pmpix_get_notify_ = ompix_get_notify_f
#        pragma weak pmpix_get_notify__ = ompix_get_notify_f

#        pragma weak PMPIX_Get_notify_f = ompix_get_notify_f
#        pragma weak PMPIX_Get_notify_f08 = ompix_get_notify_f
#    else
OMPI_GENERATE_F77_BINDINGS(PMPIX_GET_NOTIFY, pmpix_get_notify, pmpix_get_notify_,
                           pmpix_get_notify__, pompix_get_notify_f,
                           (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
                            MPI_Fint *target_rank, const MPI_Aint *target_disp,
                            MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win,
                            MPI_Fint *notification_id, MPI_Fint *ierr),
                           (origin_addr, origin_count, origin_datatype, target_rank, target_disp,
                            target_count, target_datatype, win, notification_id, ierr))
#    endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#    pragma weak MPIX_GET_NOTIFY = ompix_get_notify_f
#    pragma weak mpix_get_notify = ompix_get_notify_f
#    pragma weak mpix_get_notify_ = ompix_get_notify_f
#    pragma weak mpix_get_notify__ = ompix_get_notify_f

#    pragma weak MPIX_Get_notify_f = ompix_get_notify_f
#    pragma weak MPIX_Get_notify_f08 = ompix_get_notify_f
#else
#    if !OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS(MPIX_GET_NOTIFY, mpix_get_notify, mpix_get_notify_, mpix_get_notify__,
                           ompix_get_notify_f,
                           (char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
                            MPI_Fint *target_rank, const MPI_Aint *target_disp,
                            MPI_Fint *target_count, MPI_Fint *target_datatype, MPI_Fint *win,
                            MPI_Fint *notification_id, MPI_Fint *ierr),
                           (origin_addr, origin_count, origin_datatype, target_rank, target_disp,
                            target_count, target_datatype, win, notification_id, ierr))
#    else
#        define ompix_get_notify_f pompix_get_notify_f
#    endif
#endif

void ompix_get_notify_f(char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
                        MPI_Fint *target_rank, const MPI_Aint *target_disp, MPI_Fint *target_count,
                        MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id,
                        MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Datatype c_origin_datatype = PMPI_Type_f2c(*origin_datatype);
    MPI_Datatype c_target_datatype = PMPI_Type_f2c(*target_datatype);
    MPI_Win c_win = PMPI_Win_f2c(*win);

    c_ierr = PMPIX_Get_notify(OMPI_F2C_BOTTOM(origin_addr), OMPI_FINT_2_INT(*origin_count),
                              c_origin_datatype, OMPI_FINT_2_INT(*target_rank), *target_disp,
                              OMPI_FINT_2_INT(*target_count), c_target_datatype, c_win,
                              OMPI_FINT_2_INT(*notification_id));
    if (NULL != ierr)
        *ierr = OMPI_INT_2_FINT(c_ierr);
}
