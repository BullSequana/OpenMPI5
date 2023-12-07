
..  Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
..  Copyright 2010 Cisco Systems, Inc.  All rights reserved.
..  Copyright 2007-2008 Sun Microsystems, Inc.
..  Copyright (c) 1996 Thinking Machines Corporation

.. _mpix_win_wait_notify:


MPIX_Win_wait_notify
====================


.. include_body


NAME
----

**MPIX_Win_wait_notify**  - Waits for completion of communication identified
by the notification ID *notification_id* 

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi.h>
    int MPIX_Win_wait_notify(MPI_Win win, int notification_id)

INPUT PARAMETERS
----------------

* ``win``: Window object (handle).
* ``notification_id``: ID of the notification received from the source (positive integer).

DESCRIPTION
-----------

**MPIX_Win_wait_notify**  actively waits for the local completion of the
communication identified by the notification ID *notification_id*  on the
window *win* . When this call returns, the associated *notification_id* 
can be safely reused for other communications. If the window has not been
created with an **MPIX_Win_allocate_notify**  or **MPIX_Win_create_notify**  call,
this call returns immediately with MPI_SUCCESS.

.. note::
    The *notification_id*  parameter must be chosen inside the
    [0:MPI_MAX_NOTIFICATIONS] range. The upper bound can be changed with 'mpi_max_notification_idx' mca parameter.

ERRORS
------

Almost all MPI routines return the value of the function as an
error value.
Before the error value is returned, the current MPI error handler is called. By
default, this error handler aborts the MPI job, except for I/O function errors.
The error handler may be changed with MPI_Comm_set_errhandler; the predefined
error handler MPI_ERRORS_RETURN may be used to cause error values to be
returned. Note that MPI does not guarantee that an MPI program can continue
past an error.

.. seealso::
   * :ref:`mpix_get_notify`
   * :ref:`mpix_win_allocate_notify`
   * :ref:`mpix_win_create_notify`
   * :ref:`mpix_win_test_notify`
   * :ref:`mpix_win_wait_notify`
