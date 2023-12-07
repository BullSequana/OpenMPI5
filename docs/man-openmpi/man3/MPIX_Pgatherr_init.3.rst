
..  Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
..  Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.

.. _mpix_pgatherr_init:


MPIX_Pgatherr_init
==================


.. include_body


NAME
----

**MPIX_Pgatherr_init**  - Builds a handle for a partitioned (persistent) collective communication

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Pgatherr_init(const void *sendbuf, int spartitions, int sendcount,
                           MPI_Datatype sendtype, MPI_Request *send_requests,
                           void *recvbuf, int rpartitions, int recvcount,
                           MPI_Datatype recvtype, MPI_Request *recv_requests,
                           int root, MPI_Comm comm, MPI_Info info,
                           MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    ! or the older form: INCLUDE 'mpif.h'; INCLUDE 'mpif-ext.h'
    MPIX_PGATHERR_INIT(SENDBUF, SPARTITIONS, SENDCOUNT, SENDTYPE, SEND_REQUESTS,
                       RECVBUF, RPARTITIONS, RECVCOUNT, RECVTYPE, RECV_REQUESTS,
                       ROOT, COMM, INFO, REQUEST, IERROR)
    <type> SENDBUF(*), RECVBUF(*)
    INTEGER SPARTITIONS, SENDCOUNT, SENDTYPE, SEND_REQUESTS(*)
            INTEGER RPARTITIONS, RECVCOUNT, RECVTYPE, RECV_REQUEST(*), SROOT
    INTEGER COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Pgatherr_init(sendbuf, spartitions, sendcount,
                       sendtype, send_requests,
                       recvbuf, rpartitions, recvcount,
                       recvtype, recv_requests,
                       root, comm, info, request, ierror)
    TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
    INTEGER, INTENT(IN) :: spartitions, sendcount, rpartitions, recvcount, root
    TYPE(MPI_Datatype), INTENT(IN) :: sendtype, recvtype
    TYPE(MPI_Request) INTENT(INOUT) :: send_requests, recv_requests,
    TYPE(MPI_Comm), INTENT(IN) :: comm
    TYPE(MPI_Info), INTENT(IN) :: info
    TYPE(MPI_Request), INTENT(OUT) :: request
    INTEGER, OPTIONAL, INTENT(OUT) :: ierror

DESCRIPTION
-----------

Creates a persistent partitioned communication request for a collective operation.
Partitioned collective communications are similar to point-to-point partitioned communications: they extend persistent communication with the capability to notify MPI library of part of data that are ready the be sent or received.
MPI_Parrived can be call on any partitioned collective request as soon the request receives data.
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Pgatherr_init is functionaly equivalent to a Psend_init call for all processus in addition to N Precv_init at root.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.
In case users do not want partition request tracking, *send_requests*  or *recv_requests*  can be set to MPIX_NO_REQUESTS.

EXAMPLE
-------


.. code-block:: c

        MPI_Request req;
        MPI_Request partition_sreqs[16], partition_rreqs[16];
        int message[1024], result[1024];
        int n_part = 16, count_per_part = 42, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        if(0 == myrank) {
            MPIX_Pgatherr_init(message, n_part, count_per_part, MPI_INT, partition_sreqs,
                              result, n_part, count_per_part, MPI_INT, partition_rreqs,
                              0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        } else {
            MPIX_Pgather_init(message, n_part, count_per_part, MPI_INT, partition_sreqs
                              NULL, 0, 0, MPI_INT, MPIX_NO_REQUESTS,
                              0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        }
        MPI_Start(&req);                        // Start Psend and Precv
        for (int i = 0; i < n_part; ++i){
            MPI_Pready(i, req);             // Starts data tansfers
        }
        if (0 == myrank) {                      // Only root has initial data
            for (int i = 0; i < n_part; ++i){
                int index;
                MPI_Waitany(n_rpart, recvreqs, &index, MPI_STATUSES_IGNORE);
                /* Do compute on index part */
            }
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
