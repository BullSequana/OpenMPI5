
..  Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
..  Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.

.. _mpix_pbcastr_init:


MPIX_Pbcastr_init
=================


.. include_body


NAME
----

**MPIX_Pbcastr_init**  - Builds a handle for a partitioned (persistent) broadcast communication

SYNTAX
------


C Syntax
^^^^^^^^


.. code-block:: c

    #include <mpi-ext.h>
    int MPIX_Pbcastr_init(void *buffer, int partitions, int count,
                         MPI_Datatype datatype,  MPI_Request *p_requests,
                         int root, MPI_Comm comm, MPI_Info info, MPI_Request *request)

Fortran Syntax
^^^^^^^^^^^^^^


.. code-block:: fortran

    USE MPI_EXT
    MPIX_PBCAST_INIT(BUFFER, PARTITIONS, COUNT, DATATYPE,P_REQUESTS,
                     ROOT, COMM, INFO, REQUEST, IERROR)
    <type> BUFFER(*)
    INTEGER PARTITIONS, COUNT, DATATYPE, P_REQUESTS(*),ROOT, COMM, INFO, REQUEST, IERROR

Fortran 2008 Syntax
^^^^^^^^^^^^^^^^^^^


.. code-block:: fortran

    USE mpi_f08_ext
    MPIX_Pbcastr_init(buffer, partitions, count, datatype, root, comm, info, request,
    ierror)
    TYPE(*), DIMENSION(..), ASYNCHRONOUS :: buffer
    INTEGER, INTENT(IN) :: partitions, count, root
    TYPE(MPI_Datatype), INTENT(IN) :: datatype
    TYPE(MPI_Comm), INTENT(IN) :: comm
    TYPE(MPI_Info), INTENT(IN) :: info
    TYPE(MPI_Request), INTENT(OUT) :: request
    INTEGER, OPTIONAL, INTENT(OUT) :: ierror

DESCRIPTION
-----------

Creates a persistent partitioned communication request for a collective operation.
Partitioned collective communications are similar to point-to-point partitioned communications: they extend persistent communication with the capability to notify MPI library of part of data that are ready the be sent or received.
MPI_Parrived can be call on any partitioned collective request as soon the request receives data.
Partitioned collective communications are functionnaly equivalent to perform partitioned point-to-point communications with the same pattern as the standard collective communications would naively perform with point-to-point communications: a call to MPI_Pbcast_init is functionaly equivalent to a Precv_init call for all processus (except root) in addition to N-1 Psend_init at root.
The interface is an extension of the standard. Therefore the prefix *MPIX_*  is used instead of *MPI_*  for these request creation routines. To start, complete, and free the created request, usual MPI routines (*MPI_Start*  etc.) can be used.
On broadcast operations, calls such as MPI_Pready are only expected on the root process. Despites calls such as MPI_Parrived are only expected on non-root processes.

EXAMPLE
-------


.. code-block:: c

        MPI_Request req, partition_reqs[16];
        int message[1024];
        int n_part = 16, count_per_part = 42, myrank;
        MPIX_Pbcastr_init(message, n_part, count_per_part, MPI_INT, &partition_reqs, 0, MPI_COMM_WORLD, MPI_INFO_NULL, &req);
        MPI_Start(&req);                        // Start Psend and Precv
        if (0 == myrank) {                      // Only root has initial data
            for (int i = 0; i < n_part; ++i){
                MPI_Pready(i, req);             // Starts data tansfers
            }
        }
        for (int i = 0; i < n_part; ++i){
            int index;
            MPI_Waitany(n_rpart, recvreqs, &index, MPI_STATUSES_IGNORE);
            /* Do compute on index part */
        }
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        MPI_Request_free(&req);

.. seealso::
   * :ref:`mpi_start`
   * :ref:`mpi_startall`
   * :ref:`mpi_bcast`
   * :ref:`mpi_pready`
   * :ref:`mpi_parrived`
   * :ref:`mpi_request_free`
