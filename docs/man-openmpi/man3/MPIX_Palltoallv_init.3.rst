
..  Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.

.. _mpix_palltoallv_init:


MPIX_Palltoallv_init
====================


.. include_body


NAME
----

**MPIX_Palltoallv_init**  - Builds a handle for a partitioned (persistent) alltoallv collective communication.

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Palltoallv_init(const void *sendbuf, int sendpartitions[],
                             int sendcounts[], const int sdispls[], MPI_Datatype sendtype,
                             void *recvbuf, int recvpartitions[],
                             int recvcounts[], const int rdispls[], MPI_Datatype recvtype,
                             MPI_Comm comm, MPI_Info info,
                             MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    MPIX_ALLTOALLV_INIT(SENDBUF, SENDPARTITIONS, SENDCOUNTS, SENDTYPE,
                        RECVBUF, RECVPARTITIONS, RECVCOUNTS, RECVTYPE,
                        COMM, INFO, REQUEST, IERROR)
        <type> SENDBUF(*), RECVBUF(*)
        INTEGER SENDPARTITIONS(*), SENDCOUNTS(*), SDISPLS(*), SENDTYPE
        INTEGER RECVPARTITIONS(*), RECVCOUNTS(*), RDISPLS(*), RECVTYPE
        INTEGER COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Palltoallv_init(sendbuf, sendpartions[], sendcounts[], sdispls[], sendtype,
                        recvbuf, recvpartions[], recvcounts[], rdispls[], recvtype,
                        comm, info, request, ierror)
            TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
            TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
            INTEGER, INTENT(IN) :: sendpartions(*), sendcounts(*), sdispls(*),
            INTEGER, INTENT(IN) :: recvpartions(*), recvcounts(*), rdispls(*)
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
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Palltoallv is functionaly equivalent to a Psend_init and a Precv_init calls from and to each processus.
The i-th Psend sends the memory of *sendpartitions* [i] \* *sendcounts* [i] sendtype to rank i.
Partition indexes are accumulated: the index of rank i first partition is the sum of the i previous ranks partition count.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.

EXAMPLE
-------


.. code-block:: c

    #include "mpi.h"
    #include "mpi-ext.h"
    int main(int argc, char *argv[]) {
        MPI_Init(&argc, &argv);
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        int sdispl [2] = {0}, rdispl [2] = {0};
        MPI_Request a2a_req;
        char sbuf[10], rbuf[10];
        int spartitions[2] = {2,2}, scount[2] = {3,5};
        int rpartitions[2] = {2,1}, rcount[2] = {3,4};
        if(1 == rank) {
            spartitions [0] = 1;
            spartitions [1] = 1;
            scount [0] = 4;
            scount [1] = 8;
            rpartitions [0] = 2;
            rpartitions [1] = 1;
            rcount [0] = 5;
            rcount [1] = 8;
        }
        int total_sparts = 0, total_rparts = 0;
        for (int peer=0; peer < 2; ++ peer) {
            total_sparts += spartitions[peer];
            total_rparts += rpartitions[peer];
        }
        MPIX_Palltoallv_init(sbuf, spartitions, scount, sdispl, MPI_BYTE,
                             rbuf, rpartitions, rcount, rdispl, MPI_BYTE,
                             MPI_COMM_WORLD, MPI_INFO_NULL, &a2a_req);
        MPI_Start(&a2a_req);
        MPI_Pready_range(0, total_sparts-1, a2a_req);
        for (int part=0; part<total_rparts; ++part) {
            int arrived = 0;
            MPI_Parrived(a2a_req, part, &arrived);
            if (arrived) {
                /* Do compute on this part */
            }
        }
        MPI_Wait(&a2a_req, MPI_STATUS_IGNORE);
        MPI_Finalize();
        return 0;
    }

.. seealso::
   * :ref:`mpi_start`
   * :ref:`mpi_startall`
   * :ref:`mpi_alltoallv`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
