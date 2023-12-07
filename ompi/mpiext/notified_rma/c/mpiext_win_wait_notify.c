/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "ompi_config.h"
#include <stdio.h>

#include "ompi/mpi/c/bindings.h"
#include "ompi/runtime/params.h"
#include "ompi/communicator/communicator.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/win/win.h"
#include "ompi/mca/osc/osc.h"

#include "ompi/mpiext/notified_rma/c/mpiext_notified_rma_c.h"
#include "ompi/mpiext/notified_rma/c/mpiext_notifs_rma.h"

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPIX_Win_wait_notify = PMPIX_Win_wait_notify
#endif
#define MPIX_Win_wait_notify PMPIX_Win_wait_notify

static const char FUNC_NAME[] = "MPIX_Win_wait_notify";


// cppcheck-suppress unusedFunction
int MPIX_Win_wait_notify(MPI_Win win, int notification_id)
{
    int rc;

    if (MPI_PARAM_CHECK) {
        rc = OMPI_SUCCESS;

        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        if (ompi_win_invalid(win)) {
            return OMPI_ERRHANDLER_INVOKE(MPI_COMM_WORLD, MPI_ERR_WIN, FUNC_NAME);
        } else if (notification_id < 0 || notification_id > ompi_max_notification_idx) {
            rc = MPI_ERR_RMA_NOTIF_RANGE;
        }

        OMPI_ERRHANDLER_CHECK(rc, win, rc, FUNC_NAME);
    }

    rc = ompi_win_wait_notify(win, notification_id);
    OMPI_ERRHANDLER_RETURN(rc, win, rc, FUNC_NAME);
}
