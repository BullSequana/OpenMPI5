/*
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#ifndef MPIEXT_NOTIFS_RMA_H
#define MPIEXT_NOTIFS_RMA_H

#include "ompi_config.h"
#include "mpi.h"

#include "ompi/win/win.h"
#include "opal/class/opal_object.h"
#include "opal/class/opal_hash_table.h"
#include "opal/util/info_subscriber.h"
#include "ompi/errhandler/errhandler.h"
#include "ompi/info/info.h"
#include "ompi/communicator/communicator.h"
#include "ompi/group/group.h"
#include "ompi/mca/osc/osc.h"

int
ompi_win_create_notify(void *base, size_t size, int disp_unit,
                       ompi_communicator_t *comm, opal_info_t *info,
                       ompi_win_t **new_win);

int
ompi_win_allocate_notify(size_t size, int disp_unit, opal_info_t *info,
                         ompi_communicator_t *comm, void *baseptr,
                         ompi_win_t **newwin);

int
ompi_win_wait_notify(ompi_win_t *win, int notification_id);

int
ompi_win_test_notify(ompi_win_t *win, int notification_id, int *flag);

int
ompi_put_notify(const void *origin_addr, int origin_count,
                struct ompi_datatype_t *origin_dt,
                int target, ptrdiff_t target_disp, int target_count,
                struct ompi_datatype_t *target_dt,
                ompi_win_t *win, int notification_id);

int
ompi_get_notify(void *origin_addr, int origin_count,
                struct ompi_datatype_t *origin_dt,
                int target, ptrdiff_t target_disp, int target_count,
                struct ompi_datatype_t *target_dt,
                ompi_win_t *win, int notification_id);

#endif

