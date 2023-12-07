/*
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

#define MPIX_NO_REQUESTS ((MPI_Request*)NULL)

OMPI_DECLSPEC int MPIX_Palltoall_init(const void *sendbuf, int sendparts, int sendcount, MPI_Datatype sendtype,
                                      void *recvbuf, int recvparts, int recvcount, MPI_Datatype recvtype,
                                      MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Palltoallr_init(const void *sendbuf, int sendparts, int sendcount, MPI_Datatype sendtype, MPI_Request * sendreqs,
                                      void *recvbuf, int recvparts, int recvcount, MPI_Datatype recvtype, MPI_Request * recvreqs,
                                      MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Palltoallv_init (const void *sbuf, const int *sparts, const int *scounts, const int *sdisps, MPI_Datatype sdtype,
                                        void *rbuf, const int *rparts, const int *rcounts, const int *rdisps, MPI_Datatype rdtype,
                                        MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Palltoallvr_init (const void *sbuf, const int *sparts, const int *scounts, const int *sdisps, MPI_Datatype sdtype,
                                         MPI_Request *srequest,
                                         void *rbuf, const int *rparts, const int *rcounts, const int *rdisps, MPI_Datatype rdtype,
                                         MPI_Request *rrequest,
                                         MPI_Comm comm, MPI_Info info, MPI_Request *request);

OMPI_DECLSPEC int MPIX_Pbcast_init(void *buffer, size_t parts, size_t count, MPI_Datatype datatype,
                                   int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Pbcastr_init(void *buffer, size_t parts, size_t count, MPI_Datatype datatype, MPI_Request * reqs,
                                    int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Pgather_init(const void *sendbuf, int sendparts,
                                    int sendcount, MPI_Datatype sendtype,
                                    void *recvbuf, int recvparts,
                                    int recvcount, MPI_Datatype recvtype,
                                    int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Pgatherr_init(const void *sendbuf, int sendparts, int sendcount,
                                     MPI_Datatype sendtype, MPI_Request * sendreqs,
                                     void *recvbuf, int recvparts, int recvcount,
                                     MPI_Datatype recvtype, MPI_Request * recvreqs,
                                     int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Preduce_init(const void *sendbuf, void *recvbuf, size_t parts, int count,
                                    MPI_Datatype datatype,
                                    MPI_Op op, int root, MPI_Comm comm,
                                    MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int MPIX_Preducer_init(const void *sendbuf, void *recvbuf, size_t parts, int count,
                                     MPI_Datatype datatype,
                                     MPI_Request * sreqs, MPI_Request * rreqs,
                                     MPI_Op op, int root, MPI_Comm comm,
                                     MPI_Info info, MPI_Request *request);

/*
 * Profiling MPI API
 */

OMPI_DECLSPEC int PMPIX_Palltoall_init(const void *sendbuf, int sendparts, int sendcount, MPI_Datatype sendtype,
                                       void *recvbuf, int recvparts, int recvcount, MPI_Datatype recvtype,
                                       MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Palltoallr_init(const void *sendbuf, int sendparts, int sendcount, MPI_Datatype sendtype, MPI_Request * sendreqs,
                                        void *recvbuf, int recvparts, int recvcount, MPI_Datatype recvtype, MPI_Request * recvreqs,
                                        MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Palltoallv_init (const void *sbuf, const int *sparts, const int *scounts, const int *sdisps, MPI_Datatype sdtype,
                                        void *rbuf, const int *rparts, const int *rcounts, const int *rdisps, MPI_Datatype rdtype,
                                        MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Palltoallvr_init (const void *sbuf, const int *sparts, const int *scounts, const int *sdisps, MPI_Datatype sdtype,
                                         MPI_Request *srequest,
                                         void *rbuf, const int *rparts, const int *rcounts, const int *rdisps, MPI_Datatype rdtype,
                                         MPI_Request *rrequest,
                                         MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Pbcast_init(void *buffer, size_t parts, size_t count, MPI_Datatype datatype,
                                    int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Pbcastr_init(void *buffer, size_t parts, size_t count, MPI_Datatype datatype, MPI_Request * reqs,
                                     int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Pgather_init(const void *sendbuf, int sendparts,
                                    int sendcount, MPI_Datatype sendtype,
                                    void *recvbuf, int recvparts,
                                    int recvcount, MPI_Datatype recvtype,
                                    int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Pgatherr_init(const void *sendbuf, int sendparts, int sendcount,
                                     MPI_Datatype sendtype, MPI_Request * sendreqs,
                                     void *recvbuf, int recvparts, int recvcount,
                                     MPI_Datatype recvtype, MPI_Request * recvreqs,
                                     int root, MPI_Comm comm, MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Preduce_init(const void *sendbuf, void *recvbuf, size_t parts, int count,
                                     MPI_Datatype datatype,
                                     MPI_Op op, int root, MPI_Comm comm,
                                     MPI_Info info, MPI_Request *request);
OMPI_DECLSPEC int PMPIX_Preducer_init(const void *sendbuf, void *recvbuf, size_t parts, int count,
                                      MPI_Datatype datatype,
                                      MPI_Request * sreqs, MPI_Request * rreqs,
                                      MPI_Op op, int root, MPI_Comm comm,
                                      MPI_Info info, MPI_Request *request);
