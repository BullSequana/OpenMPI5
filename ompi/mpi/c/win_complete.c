/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
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

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Win_complete = PMPI_Win_complete
#endif
#define MPI_Win_complete PMPI_Win_complete
#endif

static const char FUNC_NAME[] = "MPI_Win_complete";


int MPI_Win_complete(MPI_Win win)
{
    int rc;

    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        if (ompi_win_invalid(win)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_WIN, FUNC_NAME);
        }
    }

    rc = win->w_osc_module->osc_complete(win);
#if OMPI_MPI_NOTIFICATIONS
    OMPI_ERRHANDLER_CHECK(rc, win, rc, FUNC_NAME);

    if (NULL != win->w_notify){
        rc = win->w_notify->w_osc_module->osc_complete(win->w_notify);
        OMPI_ERRHANDLER_RETURN(rc, win->w_notify, rc, FUNC_NAME);
    }
#endif
    OMPI_ERRHANDLER_RETURN(rc, win, rc, FUNC_NAME);
}
