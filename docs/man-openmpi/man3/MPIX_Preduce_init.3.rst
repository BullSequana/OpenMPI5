
..  Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
..  Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.

.. _mpix_preduce_init:


MPIX_Preduce_init
=================


.. include_body


NAME
----

**MPIX_Preduce_init**  - Builds a handle for a partitioned (persistent) reduction communication

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Preduce_init(const void *sendbuf, void *recvbuf,
                           int partitions, int count, MPI_Datatype datatype,
                           MPI_Op op,
                           int root, MPI_Comm comm, MPI_Info info, MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    MPI_PREDUCER_INIT(SENDBUF, RECVBUF, COUNT, DATATYPE,
                      OP, ROOT, COMM,
                      IERROR)
        <type> SENDBUF(*), RECVBUF(*),
        INTEGER PARTITIONS, COUNT, DATATYPE, OP,
                   ROOT, COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Preduce_init(sendbuf, recvbuf, partitions, count, datatype,
                       op, root, comm, info, request,
                       ierror)
    TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
    INTEGER, INTENT(IN) :: partitions, count, root
    TYPE(MPI_Datatype), INTENT(IN) :: datatype
    TYPE(MPI_Op), INTENT(IN) :: op
    TYPE(MPI_Comm), INTENT(IN) :: comm
    TYPE(MPI_Info), INTENT(IN) :: info
    TYPE(MPI_Request), INTENT(OUT) :: request
    INTEGER, OPTIONAL, INTENT(OUT) :: ierror

DESCRIPTION
-----------

Creates a persistent partitioned communication request for a collective operation.
Partitioned collective communications are similar to point-to-point partitioned communications: they extend persistent communication with the capability to notify MPI library of part of data that are ready the be sent or received.
MPI_Parrived can be call on any partitioned collective request as soon the request receives data.
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Preduce_init is functionaly equivalent to N Precv_init called at root rank (plus a data reduction for each reception) and a Psend_init to root called by all processes. Calls to MPI_Parrived are only allowed for root.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.

EXAMPLE
-------


.. code-block:: c

        MPI_Request req;
        int message[1024], result[1024];
        int n_part = 16, count_per_part = 42, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if(0 == myrank) {
            MPIX_Preduce_init(message, result, n_part, count_per_part, MPI_INT,
                              MPI_SUM, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        } else {
            MPIX_Preduce_init(message, NULL, n_part, count_per_part, MPI_INT,
                              MPI_SUM, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        }
        MPI_Start(&req);                        // Start Psend and Precv
        for (int i = 0; i < n_part; ++i){
            MPI_Pready(i, req);             // Starts data tansfers
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Request_free(&req);

.. seealso::
   * :ref:`mpi_start`
   * :ref:`mpi_startall`
   * :ref:`mpi_reduce`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
