
..  Copyright 2010 Cisco Systems, Inc.  All rights reserved.
..  Copyright 2006-2008 Sun Microsystems, Inc.
..  Copyright (c) 1996 Thinking Machines Corporation
..  Copyright 2014 Los Alamos National Security, LLC. All rights reserved.
..  Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.

.. _mpix_get_notify:


MPIX_Get_notify
===============


.. include_body


NAME
----

**MPIX_Get_notify**  - Copies data from the target memory to the origin and notifies the target.

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi.h>
    MPIX_Get_notify(void *origin_addr, int origin_count, MPI_Datatype
    origin_datatype, int target_rank, MPI_Aint target_disp,
    int target_count, MPI_Datatype target_datatype, MPI_Win win,
        int notification_id)

INPUT PARAMETERS
----------------

* ``origin_addr``: Initial address of origin buffer (choice).
* ``origin_count``: Number of entries in origin buffer (nonnegative integer).
* ``origin_datatype``: Data type of each entry in origin buffer (handle).
* ``target_rank``: Rank of target (nonnegative integer).
* ``target_disp``: Displacement from window start to the beginning of the target buffer (nonnegative integer).
* ``target_count``: Number of entries in target buffer (nonnegative integer).
* ``target datatype``: datatype of each entry in target buffer (handle)
* ``win``: window object used for communication (handle)
* ``notification_id``: ID of the notification sent to the target (positive integer).

DESCRIPTION
-----------

**MPI_Get**  copies data from the target memory to the origin, similar to **MPI_Put** , except that the direction of data transfer is reversed. The *origin_datatype*  may not specify overlapping entries in the origin buffer. The target buffer must be contained within the target window, and the copied data must fit, without truncation, in the origin buffer. Only processes within the same node can access the target window.
**MPIX_Get_notify**  is like **MPI_Get** , except that it also sends a
notification to the target with the ID *notification_id* . The remote
completion of the communication can be checked at source side with
**MPI_Win_flush** , **MPI_Win_flush_all** , **MPI_Win_unlock** , or
**MPI_Win_unlock_all**  calls. At target side, the local completion of the 
communication is performed by checking the notification ID with
**MPIX_Win_wait_notify**  or **MPIX_Win_test_notify**  calls. Notifications
are sent to the target only if both the windows at source and target have been
created with an **MPIX_Win_allocate_notify**  or **MPIX_Win_create_notify**  call.

.. note::
    The *notification_id*  parameter must be chosen inside the
    [0:MPI_MAX_NOTIFICATIONS] range. The upper bound can be changed with 'mpi_max_notification_idx' mca parameter.

ERRORS
------

Almost all MPI routines return an error value; C routines as the value of the function and Fortran routines in the last argument. C++ functions do not return errors. If the default error handler is set to MPI::ERRORS_THROW_EXCEPTIONS, then on error the C++ exception mechanism will be used to throw an MPI::Exception object.
Before the error value is returned, the current MPI error handler is
called. By default, this error handler aborts the MPI job, except for I/O function errors. The error handler may be changed with MPI_Comm_set_errhandler; the predefined error handler MPI_ERRORS_RETURN may be used to cause error values to be returned. Note that MPI does not guarantee that an MPI program can continue past an error.

.. seealso::
   * :ref:`mpix_get_notify`
   * :ref:`mpix_win_allocate_notify`
   * :ref:`mpix_win_create_notify`
   * :ref:`mpix_win_test_notify`
   * :ref:`mpix_win_wait_notify`
   * :ref:`mpi_put`
   * :ref:`mpi_get`
   * :ref:`mpi_win_flush`
   * :ref:`mpi_win_flush_all`
   * :ref:`mpi_win_unlock`
   * :ref:`mpi_win_unlock_all`
