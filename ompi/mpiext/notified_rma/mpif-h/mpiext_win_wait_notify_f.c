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
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPIX_WIN_WAIT_NOTIFY = ompix_win_wait_notify_f
#pragma weak pmpix_win_wait_notify = ompix_win_wait_notify_f
#pragma weak pmpix_win_wait_notify_ = ompix_win_wait_notify_f
#pragma weak pmpix_win_wait_notify__ = ompix_win_wait_notify_f

#pragma weak PMPIx_Win_wait_notify_f = ompix_win_wait_notify_f
#pragma weak PMPIx_Win_wait_notify_f08 = ompix_win_wait_notify_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPIX_WIN_WAIT_NOTIFY,
                           pmpix_win_wait_notify,
                           pmpix_win_wait_notify_,
                           pmpix_win_wait_notify__,
                           pompix_win_wait_notify_f,
                           (MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr),
                           (win, notification_id, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_WIN_WAIT_NOTIFY = ompix_win_wait_notify_f
#pragma weak mpix_win_wait_notify = ompix_win_wait_notify_f
#pragma weak mpix_win_wait_notify_ = ompix_win_wait_notify_f
#pragma weak mpix_win_wait_notify__ = ompix_win_wait_notify_f

#pragma weak MPIX_Win_wait_notify_f = ompix_win_wait_notify_f
#pragma weak MPIX_Win_wait_notify_f08 = ompix_win_wait_notify_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPIX_WIN_WAIT_NOTIFY,
                           mpix_win_wait_notify,
                           mpix_win_wait_notify_,
                           mpix_win_wait_notify__,
                           ompix_win_wait_notify_f,
                           (MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr),
                           (win, notification_id, ierr) )
#else
#define ompix_win_wait_notify_f pompix_win_wait_notify_f
#endif
#endif


void ompix_win_wait_notify_f(MPI_Fint *win, MPI_Fint *notification_id, MPI_Fint *ierr)
{
    int c_ierr;
    MPI_Win c_win = PMPI_Win_f2c(*win);

    c_ierr = PMPIX_Win_wait_notify(c_win, OMPI_FINT_2_INT(*notification_id));
    if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);
}
