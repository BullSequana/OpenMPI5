
..  Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
..  Copyright 2013-2014 Los Alamos National Security, LLC. All rights reserved.
..  Copyright 2010 Cisco Systems, Inc.  All rights reserved.
..  Copyright 2006-2008 Sun Microsystems, Inc.
..  Copyright (c) 1996 Thinking Machines Corporation

.. _mpix_put_notify:


MPIX_Put_notify
===============


.. include_body


NAME
----

**MPIX_Put_notify**  - Copies data from the origin memory to the target and
notifies the target.

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi.h>
    MPIX_Put_notify(const void *origin_addr, int origin_count,
    MPI_Datatype origin_datatype, int target_rank, MPI_Aint
    target_disp, int target_count, MPI_Datatype target_datatype,
    MPI_Win win, int notification_id)

INPUT PARAMETERS
----------------

* ``origin_addr``: Initial address of origin buffer (choice).
* ``origin_count``: Number of entries in origin buffer (nonnegative integer).
* ``origin_datatype``: Data type of each entry in origin buffer (handle).
* ``target_rank``: Rank of target (nonnegative integer).
* ``target_disp``: Displacement from start of window to target buffer (nonnegative integer).
* ``target_count``: Number of entries in target buffer (nonnegative integer).
* ``target_datatype``: Data type of each entry in target buffer (handle).
* ``win``: Window object used for communication (handle).
* ``notification_id``: ID of the notification sent to the target (positive integer).

DESCRIPTION
-----------

**MPIX_Put_notify**  transfers *origin_count*  successive entries of the
type specified by *origin_datatype* , starting at address *origin_addr* 
on the origin node to the target node specified by the *win* ,
*target_rank*  pair. The data are written in the target buffer at address
*target_addr*  = *window_base*  + *target_disp*  x *disp_unit* ,
where *window_base*  and *disp_unit*  are the base address and window
displacement unit specified at window initialization, by the target process.
The target buffer is specified by the arguments *target_count*  and
*target_datatype* .
The data transfer is the same as that which would occur if the origin process
executed a send operation with arguments *origin_addr* , *origin_count* ,
*origin_datatype* , *target_rank* , *tag* , *comm* , and the target
process executed a receive operation with arguments *target_addr* ,
*target_count* , *target_datatype* , *source* , *tag* , *comm* ,
where *target_addr*  is the target buffer address computed as explained
above, and *comm*  is a communicator for the group of *win* .
The communication must satisfy the same constraints as for a similar
message-passing communication. The *target_datatype*  may not specify
overlapping entries in the target buffer. The message sent must fit, without
truncation, in the target buffer. Furthermore, the target buffer must fit in
the target window. In addition, only processes within the same buffer can
access the target window.
The *target_datatype*  argument is a handle to a datatype object defined at
the origin process. However, this object is interpreted at the target process:
The outcome is as if the target datatype object were defined at the target
process, by the same sequence of calls used to define it at the origin process.
The target data type must contain only relative displacements, not absolute
addresses. The same holds for get and accumulate.
**MPIX_Put_notify**  is like **MPI_Put** , except that it also sends a
notification to the target with the ID *notification_id* . The remote
completion of the communication can be checked at source side with
**MPI_Win_flush** , **MPI_Win_flush_all** , **MPI_Win_unlock** , or
**MPI_Win_unlock_all**  calls. At target side, the local completion of the
communication is performed by checking the notification ID with
**MPIX_Win_wait_notify**  or **MPIX_Win_test_notify**  calls. Notifications
are sent to the target only if both the windows at source and target have been
created with an **MPIX_Win_allocate_notify**  **MPIX_Win_create_notify**  call.

.. note::
    The *notification_id*  parameter must be chosen inside the
    [0:MPI_MAX_NOTIFICATIONS] range. The upper bound can be changed with 'mpi_max_notification_idx' mca parameter.
    The *target_datatype*  argument is a handle to a datatype object that is
    defined at the origin process, even though it defines a data layout in the
    target process memory. This does not cause problems in a homogeneous or
    heterogeneous environment, as long as only portable data types are used
    (portable data types are defined in Section 2.4 of the MPI-2 Standard).
    The performance of a put transfer can be significantly affected, on some
    systems, from the choice of window location and the shape and location of the
    origin and target buffer: Transfers to a target window in memory allocated by
    MPI_Alloc_mem may be much faster on shared memory systems; transfers from
    contiguous buffers will be faster on most, if not all, systems; the alignment
    of the communication buffers may also impact performance.

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
   * :ref:`mpi_put`
   * :ref:`mpi_get`
   * :ref:`mpi_win_flush`
   * :ref:`mpi_win_flush_all`
   * :ref:`mpi_win_unlock`
   * :ref:`mpi_win_unlock_all`
