/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2020 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2008 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006      Cisco Systems, Inc.  All rights reserved.
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
#include "ompi/info/info.h"
#include "ompi/win/win.h"
#include "ompi/memchecker.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_Win_flush_all = PMPI_Win_flush_all
#endif
#define MPI_Win_flush_all PMPI_Win_flush_all
#endif

static const char FUNC_NAME[] = "MPI_Win_flush_all";

int MPI_Win_flush_all(MPI_Win win)
{
    int ret = MPI_SUCCESS;

    /* argument checking */
    if (MPI_PARAM_CHECK) {
        OMPI_ERR_INIT_FINALIZE(FUNC_NAME);

        if (ompi_win_invalid(win)) {
            return OMPI_ERRHANDLER_NOHANDLE_INVOKE(MPI_ERR_WIN, FUNC_NAME);
        }
        OMPI_ERRHANDLER_CHECK(ret, win, ret, FUNC_NAME);
    }

#if OMPI_MPI_NOTIFICATIONS
    if (NULL != win->w_notify){
        ret = win->w_notify->w_osc_module->osc_flush_all(win->w_notify);
        OMPI_ERRHANDLER_CHECK(ret, win->w_notify, ret, FUNC_NAME);
    }
#endif
    /* create window and return */
    ret = win->w_osc_module->osc_flush_all(win);
    OMPI_ERRHANDLER_RETURN(ret, win, ret, FUNC_NAME);
}
