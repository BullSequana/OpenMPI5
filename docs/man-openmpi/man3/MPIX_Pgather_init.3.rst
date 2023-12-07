
..  Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
..  Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.

.. _mpix_pgather_init:


MPIX_Pgather_init
=================


.. include_body


NAME
----

**MPIX_Pgather_init**  - Builds a handle for a partitioned (persistent) collective communication

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Pgather_init(const void *sendbuf, int spartitions, int sendcount,
    MPI_Datatype sendtype, void *recvbuf, int rpartitions, int recvcount,
    MPI_Datatype recvtype, int root, MPI_Comm comm, MPI_Info info,
    MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    ! or the older form: INCLUDE 'mpif.h'; INCLUDE 'mpif-ext.h'
    MPIX_PGATHER_INIT(SENDBUF, SPARTITIONS, SENDCOUNT, SENDTYPE, RECVBUF, RPARTITIONS, RECVCOUNT,
                       RECVBUF, RPARTITIONS, RECVCOUNT, RECVTYPE,
                       ROOT, COMM, INFO, REQUEST, IERROR)
    <type> SENDBUF(*), RECVBUF(*)
    INTEGER  SPARTITIONS, SENDCOUNT, SENDTYPE, RPARTITIONS, RECVCOUNT, RECVTYPE, ROOT
    INTEGER COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Pgather_init(sendbuf, spartitions, sendcount, sendtype,
                      recvbuf, rpartitions, recvcount, recvtype,
                      root, comm, info, request, ierror)
    TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
    INTEGER, INTENT(IN) :: spartitions, sendcount, rpartitions, recvcount, root
    TYPE(MPI_Datatype), INTENT(IN) :: sendtype, recvtype
    TYPE(MPI_Comm), INTENT(IN) :: comm
    TYPE(MPI_Info), INTENT(IN) :: info
    TYPE(MPI_Request), INTENT(OUT) :: request
    INTEGER, OPTIONAL, INTENT(OUT) :: ierror

DESCRIPTION
-----------

Creates a persistent partitioned communication request for a collective operation.
Partitioned collective communications are similar to point-to-point partitioned communications: they extend persistent communication with the capability to notify MPI library of part of data that are ready the be sent or received.
MPI_Parrived can be call on any partitioned collective request as soon the request receives data.
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Pgather_init is functionaly equivalent to a Psend_init call for all processus in addition to N Precv_init at root.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.

EXAMPLE
-------


.. code-block:: c

        MPI_Request req;
        int message[1024];
        int n_part = 16, count_per_part = 42, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if(0 == myrank) {
            MPIX_Pgather_init(message, n_part, count_per_part, MPI_INT,
                              result, n_part, count_per_part, MPI_INT,
                              0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        } else {
            MPIX_Pgather_init(message, n_part, count_per_part, MPI_INT,
                              NULL, 0, 0, MPI_INT,
                              0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
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
   * :ref:`mpi_gather`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
