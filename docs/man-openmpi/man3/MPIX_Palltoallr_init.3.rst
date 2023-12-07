
..  Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.

.. _mpix_palltoallr_init:


MPIX_Palltoallr_init
====================


.. include_body


NAME
----

**MPIX_Palltoallr_init**  - Builds a handle for a partitioned (persistent) alltoall collective communication with MPI request for each partition.

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Palltoallr_init(const void *sendbuf, int sendpartitions, int sendcount,
                             MPI_Datatype sendtype, MPI_Request *sendrequests,
                             void *recvbuf, int recvpartitions, int recvcount,
                             MPI_Datatype recvtype, MPI_Request *recvrequests,
                             MPI_Comm comm, MPI_Info info,
                             MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    MPIX_ALLTOALL_INIT(SENDBUF, SENDPARTITIONS, SENDCOUNT, SENDTYPE, SENDREQUESTS,
                       RECVBUF, RECVPARTITIONS, RECVCOUNT, RECVTYPE, RECVREQUESTS,
                       COMM, INFO, REQUEST, IERROR)
        <type> SENDBUF(*), RECVBUF(*)
        INTEGER SENDCOUNT, SENDPARTITIONS, SENDTYPE, RECVPARTITIONS, RECVCOUNT, RECVTYPE
        INTEGER COMM, INFO, REQUEST, SENDREQUESTS, RECVREQUESTS, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Palltoallr_init(sendbuf, sendpartions, sendcount, sendtype, sendrequests,
                        recvbuf, recvpartions, recvcount, recvrequests,
                        recvtype, comm, info, request, ierror)
    TYPE(*), DIMENSION(..), INTENT(IN), ASYNCHRONOUS :: sendbuf
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: recvbuf
    INTEGER, INTENT(IN) :: sendpartions, sendcount, recvpartions, recvcount
    TYPE(MPI_Datatype), INTENT(IN) :: sendtype, recvtype
    TYPE(MPI_Comm), INTENT(IN) :: comm
    TYPE(MPI_Info), INTENT(IN) :: info
    TYPE(MPI_Request), INTENT(OUT) :: request, sendrequests, recvrequests
    INTEGER, OPTIONAL, INTENT(OUT) :: ierror

DESCRIPTION
-----------

Creates a persistent partitioned communication request for a collective operation.
Partitioned collective communications are similar to point-to-point partitioned communications: they extend persistent communications with the capability to notify MPI library of part of data that are ready the be sent or received.
MPI_Parrived can be call on any partitioned collective request as soon the request receives data.
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPIX_Palltoallr is functionaly equivalent to a Psend_init and a Precv_init calls from and to each processus.
MPIX_Palltoallr is an extension of MPIX_Palltoall, that fills users arrays *sendrequests*  and *recvrequests*  with MPI_request to let them track partitions completion with MPI rountines such as MPI_Waitany or MPI_Testany. This approach targets task based applications that only needs a part of the data to unlock compute phases.
In case some of these requests are not needed, *sendrequests*  or *recvrequests*  can be set to MPIX_NO_REQUESTS.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.

EXAMPLE
-------


.. code-block:: c

    #include <stdlib.h>
    #include <mpi.h>
    #include <mpi-ext.h>
    int main(int argc, char*argv[]){
        MPI_Request req;
        MPI_Init(&argc, &argv);
        int n_spart = 4, count_per_spart = 42;
        int n_rpart = 2, count_per_rpart = 84;
        int commsize;
        MPI_Comm_size(MPI_COMM_WORLD, &commsize);
        MPI_Request *recvreqs = malloc(n_rpart*commsize * sizeof(MPI_Request));
        int total_spart = n_spart * commsize;
        int total_rpart = n_rpart * commsize;
        double *send_message = malloc(total_spart*count_per_spart*sizeof(double));
        double *recv_message = malloc(total_rpart*count_per_rpart*sizeof(double));
        MPIX_Palltoallr_init(send_message, n_spart, count_per_spart, MPI_DOUBLE, MPIX_NO_REQUESTS,
                             recv_message, n_rpart, count_per_rpart, MPI_DOUBLE, recvreqs,
                             MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        MPI_Start(&req);
        for (int i = 0; i < total_spart; ++i){
            MPI_Pready(i, req);             // Starts data transfer
        }
        for (int i = 0; i < total_rpart; ++i){
            int index;
            MPI_Waitany(n_rpart, recvreqs, &index, MPI_STATUSES_IGNORE);
            /* Do compute on index part */
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Request_free(&req);
        free(recvreqs);
        free(send_message);
        free(recv_message);
        MPI_Finalize();
    }

.. seealso::
   * :ref:`mpi_start`
   * :ref:`mpi_startall`
   * :ref:`mpi_alltoall`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
