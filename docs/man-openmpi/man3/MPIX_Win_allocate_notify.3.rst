
..  Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
..  Copyright 2015      Los Alamos National Security, LLC. All rights reserved.
..  Copyright 2010 Cisco Systems, Inc.  All rights reserved.
..  Copyright 2007-2008 Sun Microsystems, Inc.
..  Copyright (c) 1996 Thinking Machines Corporation

.. _mpix_win_allocate_notify:


MPIX_Win_allocate_notify
========================


.. include_body


NAME
----

**MPIX_Win_allocate_notify**  - One-sided MPI notifications call that
allocates memory and returns a window object for RMA operations with
notifications support.

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi.h>
    int MPIX_Win_allocate_notify (MPI_Aint size, int disp_unit,
                MPI_Info info, MPI_Comm comm, void *baseptr,
                MPI_Win *win)

INPUT PARAMETERS
----------------

* ``size``: Size of window in bytes (nonnegative integer).
* ``disp_unit``: Local unit size for displacements, in bytes (positive integer).
* ``info``: Info argument (handle).
* ``comm``: Communicator (handle).

OUTPUT PARAMETERS
-----------------

* ``baseptr``: Initial address of window.
* ``win``: Window object returned by the call (handle).

DESCRIPTION
-----------

**MPIX_Win_allocate_notify**  is a collective call executed by all processes
in the group of *comm* . On each process, it allocates memory of at least
*size*  bytes, returns a pointer to it, and returns a window object that can
be used by all processes in *comm*  to perform RMA operations with or
without notified communications support. The returned memory consists of
*size*  bytes local to each process, starting at address *baseptr*  and
is associated with the window as if the user called **MPIX_Win_create_notify**  on
existing memory. The *size*  argument may be different at each process and
*size*  = 0 is valid; however, a library might allocate and expose more
memory in order to create a fast, globally symmetric allocation. The discussion
of and rationales for **MPI_Alloc_mem**  and **MPI_Free_mem**  in MPI-3.1
\[char167] 8.2 also apply to **MPIX_Win_allocate_notify** ; in particular, see the
rationale in MPI-3.1 \[char167] 8.2 for an explanation of the type used for
*baseptr* .
The displacement unit argument is provided to facilitate address arithmetic in
RMA operations: the target displacement argument of an RMA operation is scaled
by the factor *disp_unit*  specified by the target process, at window
creation.
For supported info keys see **MPI_Win_create** .
**MPIX_Win_allocate_notify**  extends the **MPI_Win_allocate**  call by
allowing processes to communicate with notified communications. For details on
notified communications see **MPIX_Put_notify** .

.. note::
    Common choices for *disp_unit*  are 1 (no scaling), and *sizeof(type)* ,
    for a window that consists of an array of elements of type *type* . The
    later choice will allow one to use array indices in RMA calls, and have those
    scaled correctly to byte displacements, even in a heterogeneous environment.

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
   * :ref:`mpi_alloc_mem`
   * :ref:`mpi_free_mem`
   * :ref:`mpi_win_create`
   * :ref:`mpi_win_allocate`
   * :ref:`mpi_win_allocate_shared`
