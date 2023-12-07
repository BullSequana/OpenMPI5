
..  Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.

.. _mpix_palltoall_init:


MPIX_Palltoall_init
===================


.. include_body


NAME
----

**MPIX_Palltoall_init**  - Builds a handle for a partitioned (persistent) alltoall collective communication

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Palltoall_init(const void *sendbuf, int sendpartitions, int sendcount,
    MPI_Datatype sendtype, void *recvbuf, int recvpartitions, int recvcount,
    MPI_Datatype recvtype, MPI_Comm comm, MPI_Info info,
    MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    MPIX_ALLTOALL_INIT(SENDBUF, SENDPARTITIONS, SENDCOUNT, SENDTYPE,
                       RECVBUF, RECVPARTITIONS, RECVCOUNT, RECVTYPE,
                       COMM, INFO, REQUEST, IERROR)
        <type> SENDBUF(*), RECVBUF(*)
        INTEGER SENDCOUNT, SENDPARTITIONS, SENDTYPE, RECVPARTITIONS, RECVCOUNT, RECVTYPE
        INTEGER COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Palltoall_init(sendbuf, sendpartions, sendcount, sendtype,
                        recvbuf, recvpartions, recvcount, recvtype,
                        comm, info, request, ierror)
    TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
    INTEGER, INTENT(IN) :: sendpartions, sendcount, recvpartions, recvcount
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
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Palltoall is functionaly equivalent to a Psend_init and a Precv_init calls from and to each processus.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.

EXAMPLE
-------


.. code-block:: c

        MPI_Request req;
        int send_message[1024], recv_message[1024];
        int n_spart = 16, count_per_spart = 42;
        int n_rpart = 8, count_per_rpart = 84;
        MPIX_Palltoall_init(send_message, n_spart, count_per_spart, MPI_DOUBLE,
                            recv_message, n_rpart, count_per_rpart, MPI_DOUBLE,
                            MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        MPI_Start(&req);
        for (int i = 0; i < n_spart; ++i){
            MPI_Pready(i, req);             // Starts data transfer
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Request_free(&req);

.. seealso::
   * :ref:`mpi_start`
   * :ref:`mpi_startall`
   * :ref:`mpi_alltoall`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
