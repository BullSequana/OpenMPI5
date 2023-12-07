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
#        pragma weak PMPIX_WIN_CREATE_NOTIFY = ompix_win_create_notify_f
#        pragma weak pmpix_win_create_notify = ompix_win_create_notify_f
#        pragma weak pmpix_win_create_notify_ = ompix_win_create_notify_f
#        pragma weak pmpix_win_create_notify__ = ompix_win_create_notify_f

#        pragma weak PMPIX_Win_create_notify_f = ompix_win_create_notify_f
#        pragma weak PMPIX_Win_create_notify_f08 = ompix_win_create_notify_f
#    else
OMPI_GENERATE_F77_BINDINGS(PMPIX_WIN_CREATE_NOTIFY, pmpix_win_create_notify,
                           pmpix_win_create_notify_, pmpix_win_create_notify__,
                           pompix_win_create_notify_f,
                           (char *base, const MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info,
                            MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr),
                           (base, size, disp_unit, info, comm, win, ierr))
#    endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#    pragma weak MPIX_WIN_CREATE_NOTIFY = ompix_win_create_notify_f
#    pragma weak mpix_win_create_notify = ompix_win_create_notify_f
#    pragma weak mpix_win_create_notify_ = ompix_win_create_notify_f
#    pragma weak mpix_win_create_notify__ = ompix_win_create_notify_f

#    pragma weak MPIX_Win_create_notify_f = ompix_win_create_notify_f
#    pragma weak MPIX_Win_create_notify_f08 = ompix_win_create_notify_f
#else
#    if !OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS(MPIX_WIN_CREATE_NOTIFY, mpix_win_create_notify, mpix_win_create_notify_,
                           mpix_win_create_notify__, ompix_win_create_notify_f,
                           (char *base, const MPI_Aint *size, MPI_Fint *disp_unit, MPI_Fint *info,
                            MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr),
                           (base, size, disp_unit, info, comm, win, ierr))
#    else
#        define ompix_win_create_notify_f pompix_win_create_notify_f
#    endif
#endif

void ompix_win_create_notify_f(char *base, const MPI_Aint *size, MPI_Fint *disp_unit,
                               MPI_Fint *info, MPI_Fint *comm, MPI_Fint *win, MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Win c_win;
    MPI_Info c_info;
    MPI_Comm c_comm;

    c_comm = PMPI_Comm_f2c(*comm);
    c_info = PMPI_Info_f2c(*info);

    c_ierr = PMPIX_Win_create_notify(base, *size, OMPI_FINT_2_INT(*disp_unit), c_info, c_comm,
                                     &c_win);
    if (NULL != ierr)
        *ierr = OMPI_INT_2_FINT(c_ierr);

    if (MPI_SUCCESS == c_ierr) {
        *win = PMPI_Win_c2f(c_win);
    }
}
