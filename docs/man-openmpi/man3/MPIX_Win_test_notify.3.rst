
..  Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
..  Copyright 2010 Cisco Systems, Inc.  All rights reserved.
..  Copyright 2007-2008 Sun Microsystems, Inc.
..  Copyright (c) 1996 Thinking Machines Corporation

.. _mpix_win_test_notify:


MPIX_Win_test_notify
====================


.. include_body


NAME
----

**MPIX_Win_test_notify**  - Tests for completion of communication identified
by the notification ID *notification_id* 

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi.h>
    int MPIX_Win_test_notify(MPI_Win win, int notification_id,
                int *flag)

INPUT PARAMETERS
----------------

* ``win``: Window object (handle).
* ``notification_id``: ID of the notification received from the source (positive integer).

OUTPUT PARAMETERS
-----------------

* ``flag``: The returning state of the test for communication completion.

DESCRIPTION
-----------

**MPIX_Win_test_notify**  is a nonblocking version of
**MPIX_Win_wait_notify** . It returns *flag = true*  if
**MPIX_Win_wait_notify**  would return, *flag = false*  otherwise. The
effect of return of MPIX_Win_test_notify with *flag = true*  is the same as
the effect of a return of **MPIX_Win_wait_notify** . If *flag = false*  is
returned, then the call has no visible effect.
Invoke **MPIX_Win_test_notify**  only where **MPIX_Win_wait_notify**  can be
invoked.  Once the call has returned *flag = true* , it must not be invoked
anew, until a new communication with the same *notification_id*  is posted.

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
