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
#        pragma weak PMPIX_PUT_NOTIFY = ompix_put_notify_f
#        pragma weak pmpix_put_notify = ompix_put_notify_f
#        pragma weak pmpix_put_notify_ = ompix_put_notify_f
#        pragma weak pmpix_put_notify__ = ompix_put_notify_f

#        pragma weak PMPIX_Put_notify_f = ompix_put_notify_f
#        pragma weak PMPIX_Put_notify_f08 = ompix_put_notify_f
#    else
OMPI_GENERATE_F77_BINDINGS(
    PMPIX_PUT_NOTIFY, pmpix_put_notify, pmpix_put_notify, pmpix_put_notify__, pompix_put_notify_f,
    (const char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
     MPI_Fint *target_rank, const MPI_Aint *target_disp, MPI_Fint *target_count,
     MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr),
    (origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
     target_datatype, win, notification_id, ierr))
#    endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#    pragma weak MPIX_PUT_NOTIFY = ompix_put_notify_f
#    pragma weak mpix_put_notify = ompix_put_notify_f
#    pragma weak mpix_put_notify_ = ompix_put_notify_f
#    pragma weak mpix_put_notify__ = ompix_put_notify_f

#    pragma weak MPIX_Put_notify_f = ompix_put_notify_f
#    pragma weak MPIX_Put_notify_f08 = ompix_put_notify_f
#else
#    if !OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS(
    MPIX_PUT_NOTIFY, mpix_put_notify, mpix_put_notify_, mpix_put_notify__, ompix_put_notify_f,
    (const char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
     MPI_Fint *target_rank, const MPI_Aint *target_disp, MPI_Fint *target_count,
     MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr),
    (origin_addr, origin_count, origin_datatype, target_rank, target_disp, target_count,
     target_datatype, win, notification_id, ierr))
#    else
#        define ompix_put_notify_f pompix_put_notify_f
#    endif
#endif

void ompix_put_notify_f(const char *origin_addr, MPI_Fint *origin_count, MPI_Fint *origin_datatype,
                        MPI_Fint *target_rank, const MPI_Aint *target_disp, MPI_Fint *target_count,
                        MPI_Fint *target_datatype, MPI_Fint *win, MPI_Fint *notification_id,
                        MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Datatype c_origin_datatype = PMPI_Type_f2c(*origin_datatype);
    MPI_Datatype c_target_datatype = PMPI_Type_f2c(*target_datatype);
    MPI_Win c_win = PMPI_Win_f2c(*win);

    c_ierr = PMPIX_Put_notify(OMPI_F2C_BOTTOM(origin_addr), OMPI_FINT_2_INT(*origin_count),
                              c_origin_datatype, OMPI_FINT_2_INT(*target_rank), *target_disp,
                              OMPI_FINT_2_INT(*target_count), c_target_datatype, c_win,
                              OMPI_FINT_2_INT(*notification_id));
    if (NULL != ierr)
        *ierr = OMPI_INT_2_FINT(c_ierr);
}
