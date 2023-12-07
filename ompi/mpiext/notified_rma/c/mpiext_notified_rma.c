/*
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#include "ompi_config.h"

#include <stdio.h>
#include <string.h>

#include "opal/util/show_help.h"
#include "opal/constants.h"
#include "ompi/runtime/params.h"
#include "ompi/win/win.h"
#include "ompi/attribute/attribute.h"
#include "ompi/mpiext/notified_rma/c/mpiext_notified_rma_c.h"
#include "ompi/mpiext/notified_rma/c/mpiext_notifs_rma.h"

int
ompi_win_create_notify(void *base, size_t size, int disp_unit,
                       ompi_communicator_t *comm, opal_info_t *info,
                       ompi_win_t **newwin)
{
    int ret;
    void *base_notify;
    size_t size_notify;
    ompi_win_t *win, *w_notify;

    size_notify = (ompi_max_notification_idx + 1) * sizeof(int);

    /* Allocate notification window */
    ret = ompi_win_allocate(size_notify, sizeof(int), info, comm,
                            &base_notify, &w_notify);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    memset(base_notify, 0, size_notify);

    /* Create the main window */
    ret = ompi_win_create(base, size, disp_unit, comm, info, &win);
    if (OMPI_SUCCESS != ret) {
        ompi_win_free(w_notify);
        return ret;
    }

    win->w_notify = w_notify;
    win->w_notify->is_notif_window = 1;
    win->notif_base_pointer = base_notify;

    *newwin = win;

    return OMPI_SUCCESS;
}

int
ompi_win_allocate_notify(size_t size, int disp_unit, opal_info_t *info,
                         ompi_communicator_t *comm, void *baseptr,
                         ompi_win_t **newwin)
{
    int ret;
    void *base;
    void *base_notify;
    size_t size_notify;
    ompi_win_t *win, *w_notify;

    size_notify = (ompi_max_notification_idx + 1) * sizeof(int);

    /* Allocate notification window */
    ret = ompi_win_allocate(size_notify, sizeof(int), info, comm,
                            &base_notify, &w_notify);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    memset(base_notify, 0, size_notify);

    /* Allocate the main window */
    ret = ompi_win_allocate(size, disp_unit, info, comm, &base, &win);
    if (OMPI_SUCCESS != ret) {
        ompi_win_free(w_notify);
        return ret;
    }

    win->w_notify = w_notify;
    win->w_notify->is_notif_window = 1;
    win->notif_base_pointer = base_notify;

    *((void**) baseptr) = base;
    *newwin = win;

    return OMPI_SUCCESS;
}

int
ompi_win_wait_notify(ompi_win_t *win, int notification_id)
{
    int *model, model_found, ret;

    if (NULL == win->w_notify) {
            return MPI_ERR_WIN;
    }
    ret = ompi_attr_get_c(win->w_notify->w_keyhash, MPI_WIN_MODEL,
                          (void**)&model, &model_found);
    OMPI_ERRHANDLER_CHECK(ret, win, MPI_ERR_OTHER, __func__);

    int *notif_ptr = (int*) win->notif_base_pointer + notification_id;
    do {
        int value;
        /* atomic cmp_swap stores current value in 'old_value' */
        int old_value = 1;
        if(!model_found || MPI_WIN_SEPARATE == *model){
            /* Update local vision of process exposed memory with a MPI_Win_sync
             * directly called by the osc module in charge the notification window */
            win->w_notify->w_osc_module->osc_sync(win->w_notify);
        }
        value = opal_atomic_compare_exchange_strong_32(notif_ptr, &old_value, 0);
        if (value){
            break;
        }

        /* The remote is maybe waiting for one of our pending
         * communication before sending its notification */
        opal_progress();
    } while(1);

    return OMPI_SUCCESS;
}

int
ompi_win_test_notify(ompi_win_t *win, int notification_id, int *flag)
{
    int *model, model_found, ret;

    if (NULL == win->w_notify) {
            return MPI_ERR_WIN;
    }

    ret = ompi_attr_get_c(win->w_notify->w_keyhash, MPI_WIN_MODEL,
                          (void**)&model, &model_found);
    OMPI_ERRHANDLER_CHECK(ret, win->w_notify, MPI_ERR_OTHER, __func__);

    int *notif_ptr = (int*) win->notif_base_pointer + notification_id;
    /* atomic cmp_swap stores current value in 'old_value' */
    int old_value = 1;
    if(!model_found || MPI_WIN_SEPARATE == *model){
        /* Update local vision of process exposed memory with a MPI_Win_sync
         * directly called by the osc module in charge the notification window */
        win->w_notify->w_osc_module->osc_sync(win->w_notify);
    }
    *flag = opal_atomic_compare_exchange_strong_32(notif_ptr, &old_value, 0);

    if (! *flag){
        /* The remote is maybe waiting for one of our pending
         * communication before sending its notification */
        opal_progress();
    }

    return OMPI_SUCCESS;
}

int
ompi_put_notify(const void *origin_addr, int origin_count,
                struct ompi_datatype_t *origin_dt,
                int target, ptrdiff_t target_disp, int target_count,
                struct ompi_datatype_t *target_dt,
                ompi_win_t *win, int notification_id)
{
    if (NULL == win->w_notify) {
        return MPI_ERR_WIN;
    }

    if (win->w_osc_module->osc_put_notify){
        return win->w_osc_module->osc_put_notify(origin_addr, origin_count, origin_dt,
                                                 target, target_disp, target_count, target_dt,
                                                 win, notification_id);
    } /* else fallback on basic put - flush - put */

    int ret;

    /* Put the data */
    ret = win->w_osc_module->osc_put(origin_addr, origin_count, origin_dt,
                                     target, target_disp, target_count,
                                     target_dt, win);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    /* Fetch the main window to ensure put is ended */
    ret = win->w_osc_module->osc_flush(target, win);
    if (OMPI_SUCCESS != ret) {
        if (OMPI_ERR_RMA_SYNC == ret){
            opal_show_help("help-mpi-notifications.txt",
                           "flush_err_rma_sync", true, "MPI_Put_notify");
        }
        return ret;
    }

    int notification = 1;

    /* Notify the target through a put on the notification window */
    return win->w_notify->w_osc_module->osc_put(&notification, 1, MPI_INT,
                                                target, notification_id, 1,
                                                MPI_INT, win->w_notify);

}

int
ompi_get_notify(void *origin_addr, int origin_count,
                struct ompi_datatype_t *origin_dt,
                int target, ptrdiff_t target_disp, int target_count,
                struct ompi_datatype_t *target_dt,
                ompi_win_t *win, int notification_id)
{
    if (NULL == win->w_notify) {
        return MPI_ERR_WIN;
    }
    if (win->w_osc_module->osc_get_notify){
        return win->w_osc_module->osc_get_notify(origin_addr, origin_count, origin_dt,
                                                 target, target_disp, target_count, target_dt,
                                                 win, notification_id);
    } /* else fallback on basic get - flush - put */

    int ret;

    /* Get the data */
    ret = win->w_osc_module->osc_get(origin_addr, origin_count, origin_dt,
                                     target, target_disp, target_count,
                                     target_dt, win);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    /* Fetch the main window to ensure get is ended */
    ret = win->w_osc_module->osc_flush(target, win);
    if (OMPI_SUCCESS != ret) {
        if (OMPI_ERR_RMA_SYNC == ret){
            opal_show_help("help-mpi-notifications.txt",
                           "flush_err_rma_sync", true, "MPI_Get_notify");
        }
        return ret;
    }

    int notification = 1;

    /* Notify the target through a put on the notification window */
    return win->w_notify->w_osc_module->osc_put(&notification, 1, MPI_INT,
                                                target, notification_id, 1,
                                                MPI_INT, win->w_notify);
}

