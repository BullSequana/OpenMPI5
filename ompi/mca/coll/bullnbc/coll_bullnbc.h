/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2013 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2008      Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2015 Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2014-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2016-2017 IBM Corporation.  All rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BULLNBC_EXPORT_H
#define MCA_COLL_BULLNBC_EXPORT_H

#include "ompi/mca/coll/coll.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "opal/sys/atomic.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_dynamic_rules.h"

BEGIN_C_DECLS

/*********************** BullNBC tuning parameters ************************/


#define NBC_SCHED_DICT_UPPER 1024 /* max. number of dict entries */
#define NBC_SCHED_DICT_LOWER 512  /* nuber of dict entries after wipe, if SCHED_DICT_UPPER is reached */

/********************* end of BullNBC tuning parameters ************************/

/* Function return codes  */
#define NBC_SUCCESS                     OMPI_SUCCESS
#define NBC_OK                          NBC_SUCCESS
#define NBC_OOR                         OMPI_ERR_OUT_OF_RESOURCE
#define NBC_BAD_SCHED                   OMPI_ERR_FATAL /* bad schedule */
#define NBC_CONTINUE                    OMPI_ERR_INTERUPTED /* progress not done */
#define NBC_DATATYPE_NOT_SUPPORTED      OMPI_ERR_UNKNOWN_DATA_TYPE /* datatype not supported or not valid */
#define NBC_OP_NOT_SUPPORTED            OMPI_ERR_NOT_SUPPORTED /* operation not supported or not valid */
#define NBC_NOT_IMPLEMENTED             OMPI_ERR_NOT_IMPLEMENTED
#define NBC_INVALID_PARAM               OMPI_ERR_BAD_PARAM /* invalid parameters */

/* number of implemented collective functions */
#define NBC_NUM_COLL 17

/* Hint if non-blocking collectives are likely to use GPU buffers */
#define GPU_LIKELYHOOD OPAL_UNLIKELY

extern bool bullnbc_ibcast_skip_dt_decision;

/* forced algorithm choices */
/* this structure is for storing the indexes to the forced algorithm mca params... */
/* we get these at component query (so that registered values appear in ompi_info) */
struct coll_bullnbc_force_algorithm_mca_param_indices_t {
    int algorithm; /* which algorithm you want to force */
    int algorithm_count; /* Number of available algorithms */
    int segsize;
    /* TODO : Add fanout, max_requests */
};
typedef struct coll_bullnbc_force_algorithm_mca_param_indices_t coll_bullnbc_force_algorithm_mca_param_indices_t;

struct ompi_coll_bullnbc_component_t {
    mca_coll_base_component_2_4_0_t super;
    opal_free_list_t requests;
    opal_list_t active_requests;
    opal_free_list_t schedules;
    int32_t active_comms;
    opal_mutex_t lock;                /* protect access to the active_requests list */
    bool use_dynamic_rules;
    bool reduce_allow_non_commutative_support;
    int dynamic_rules_verbose;
    /* the debug level */
    int debug_verbose;
    int stream;
    coll_bullnbc_force_algorithm_mca_param_indices_t forced_params[COLLCOUNT];
    /* cached decision table stuff */
    ompi_coll_base_alg_rule_t *all_base_rules;
    int dynamic_rules_fileformat;
    char* dynamic_rules_filename;
    int debug_read_user_data;
};
typedef struct ompi_coll_bullnbc_component_t ompi_coll_bullnbc_component_t;

/* Globally exported variables */
OMPI_DECLSPEC extern ompi_coll_bullnbc_component_t mca_coll_bullnbc_component;

struct ompi_coll_bullnbc_module_t {
    mca_coll_base_module_t super;
    opal_mutex_t mutex;
    int32_t comm_registered;
    int tag;
    /* the communicator rules for each MPI collective for ONLY my comsize */
    ompi_coll_base_com_rule_t *com_rules[COLLCOUNT];
};
typedef struct ompi_coll_bullnbc_module_t ompi_coll_bullnbc_module_t;
OBJ_CLASS_DECLARATION(ompi_coll_bullnbc_module_t);

typedef ompi_coll_bullnbc_module_t NBC_Comminfo;

struct BULLNBC_Schedule {
    opal_free_list_item_t super;
    MPI_Request *requests;

    int32_t num_rounds;
    int32_t max_rounds;

    int32_t max_cmds;
    int32_t dynamic;

    int32_t num_cmds;
    int32_t max_round_arity;

    int *arity;
    int *rounds;
    void *cmds; /* break include */
};
typedef struct BULLNBC_Schedule BULLNBC_Schedule;
OBJ_CLASS_DECLARATION(BULLNBC_Schedule);

struct ompi_coll_bullnbc_request_t {
    ompi_coll_base_nbc_request_t super;
    MPI_Comm comm;
    long row_offset;
    int max_round;
    int cur_round;
    bool nbc_complete; /* status in bullnbc level */
    int tag;
    volatile int req_count;
    ompi_request_t **req_array;
    NBC_Comminfo *comminfo;
    BULLNBC_Schedule *schedule;
    void *tmpbuf; /* temporary buffer e.g. used for Reduce */
    /* TODO: we should make a handle pointer to a state later (that the user
     * can move request handles) */
};
typedef struct ompi_coll_bullnbc_request_t ompi_coll_bullnbc_request_t;
OBJ_CLASS_DECLARATION(ompi_coll_bullnbc_request_t);

typedef ompi_coll_bullnbc_request_t NBC_Handle;


#define OMPI_COLL_BULLNBC_REQUEST_ALLOC(comm, persistent, req)           \
    do {                                                                \
        opal_free_list_item_t *item;                                    \
        item = opal_free_list_wait (&mca_coll_bullnbc_component.requests); \
        req = (ompi_coll_bullnbc_request_t*) item;                       \
        OMPI_REQUEST_INIT(&req->super.super, persistent);               \
        req->super.super.req_mpi_object.comm = comm;                    \
        req->cur_round = 0;                                             \
    } while (0)

#define OMPI_COLL_BULLNBC_REQUEST_RETURN(req)                            \
    do {                                                                \
        OMPI_REQUEST_FINI(&(req)->super.super);                         \
        opal_free_list_return (&mca_coll_bullnbc_component.requests,     \
                               (opal_free_list_item_t*) (req));         \
    } while (0)

int ompi_coll_bullnbc_progress(void);

int BULLNBC_Init_comm(MPI_Comm comm, ompi_coll_bullnbc_module_t *module);
#define NBC_Progress BULLNBC_Progress
int NBC_Progress(NBC_Handle *handle);

int ompi_coll_bullnbc_allgather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_allgatherv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_allreduce_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_alltoall_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_alltoallv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_alltoallw_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_barrier_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_bcast_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_exscan_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_gather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_gatherv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_gatherw_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_reduce_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_reduce_scatter_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_reduce_scatter_block_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_scan_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_scatter_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_scatterv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_neighbor_allgather_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_neighbor_allgatherv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_neighbor_alltoall_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_neighbor_alltoallv_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);
int ompi_coll_bullnbc_neighbor_alltoallw_check_forced_init (coll_bullnbc_force_algorithm_mca_param_indices_t *mca_param_indices);

int ompi_coll_bullnbc_iallgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iallgatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iallreduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoall(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                               MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoallv(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoallw(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                struct ompi_datatype_t * const *sendtypes, void* recvbuf, const int *recvcounts, const int *rdispls,
                                struct ompi_datatype_t * const *recvtypes, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ibarrier(struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ibcast(void *buffer, int count, MPI_Datatype datatype, int root,
                            struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iexscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                             struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                             MPI_Datatype recvtype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igatherv(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                              int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igatherw(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                              int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                             MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce_scatter(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                     MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce_scatter_block(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                           MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                           struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iscan(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                           struct ompi_communicator_t *comm, ompi_request_t ** request,
                           struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iscatter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                              struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iscatterv(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                               void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module);


int ompi_coll_bullnbc_iallgather_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iallgatherv_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                 MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iallreduce_inter(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoall_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                               MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoallv_inter(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                MPI_Datatype recvtype, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ialltoallw_inter(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                      struct ompi_datatype_t * const *sendtypes, void* recvbuf, const int *recvcounts, const int *rdispls,
                                      struct ompi_datatype_t * const *recvtypes, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ibarrier_inter(struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ibcast_inter(void *buffer, int count, MPI_Datatype datatype, int root,
                            struct ompi_communicator_t *comm, ompi_request_t ** request,
                            struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igather_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                             MPI_Datatype recvtype, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igatherv_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                              int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_igatherw_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                              int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce_inter(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                             MPI_Op op, int root, struct ompi_communicator_t *comm, ompi_request_t ** request,
                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce_scatter_inter(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                     MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ireduce_scatter_block_inter(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                                 MPI_Op op, struct ompi_communicator_t *comm, ompi_request_t ** request,
                                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iscatter_inter(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                              void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                              struct ompi_communicator_t *comm, ompi_request_t ** request,
                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_iscatterv_inter(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                               void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                               struct ompi_communicator_t *comm, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module);


int ompi_coll_bullnbc_ineighbor_allgather(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                         int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                         ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ineighbor_allgatherv(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                          const int *rcounts, const int *displs, MPI_Datatype rtype,
                                          struct ompi_communicator_t *comm, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ineighbor_alltoall(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                        int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                        ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ineighbor_alltoallv(const void *sbuf, const int *scounts, const int *sdispls, MPI_Datatype stype,
                                         void *rbuf, const int *rcounts, const int *rdispls, MPI_Datatype rtype,
                                         struct ompi_communicator_t *comm, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_ineighbor_alltoallw(const void *sbuf, const int *scounts, const MPI_Aint *sdisps, struct ompi_datatype_t * const *stypes,
                                         void *rbuf, const int *rcounts, const MPI_Aint *rdisps, struct ompi_datatype_t * const *rtypes,
                                         struct ompi_communicator_t *comm, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);

int ompi_coll_bullnbc_allgather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_allgatherv_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                     MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_allreduce_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                    struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoall_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                   MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoallv_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                    MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                    MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoallw_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                    struct ompi_datatype_t * const *sendtypes, void* recvbuf, const int *recvcounts, const int *rdispls,
                                    struct ompi_datatype_t * const *recvtypes, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_barrier_init(struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_bcast_init(void *buffer, int count, MPI_Datatype datatype, int root,
                                struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_exscan_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                 struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gather_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                 MPI_Datatype recvtype, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gatherv_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                                  int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gatherw_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                  int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                 MPI_Op op, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_scatter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                         MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info,  ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_scatter_block_init(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                               MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_scan_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                               struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                               struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_scatter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                  void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                  struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_scatterv_init(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                   void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                   struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module);

int ompi_coll_bullnbc_allgather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_allgatherv_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *displs,
                                           MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                           struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_allreduce_inter_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op,
                                          struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoall_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                         MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoallv_inter_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                          MPI_Datatype sendtype, void* recvbuf, const int *recvcounts, const int *rdispls,
                                          MPI_Datatype recvtype, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_alltoallw_inter_init(const void* sendbuf, const int *sendcounts, const int *sdispls,
                                          struct ompi_datatype_t * const *sendtypes, void* recvbuf, const int *recvcounts, const int *rdispls,
                                          struct ompi_datatype_t * const *recvtypes, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                          struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_barrier_inter_init(struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_bcast_inter_init(void *buffer, int count, MPI_Datatype datatype, int root,
                                      struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gather_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, int recvcount,
                                       MPI_Datatype recvtype, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gatherv_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                        void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype recvtype,
                                        int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_gatherw_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                        void* recvbuf, const int *recvcounts, const int *displs, MPI_Datatype *recvtypes,
                                        int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_inter_init(const void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype,
                                       MPI_Op op, int root, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                       struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_scatter_inter_init(const void* sendbuf, void* recvbuf, const int *recvcounts, MPI_Datatype datatype,
                                               MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                               struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_reduce_scatter_block_inter_init(const void* sendbuf, void* recvbuf, int recvcount, MPI_Datatype datatype,
                                                     MPI_Op op, struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                                     struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_scatter_inter_init(const void* sendbuf, int sendcount, MPI_Datatype sendtype,
                                        void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                        struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                        struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_scatterv_inter_init(const void* sendbuf, const int *sendcounts, const int *displs, MPI_Datatype sendtype,
                                         void* recvbuf, int recvcount, MPI_Datatype recvtype, int root,
                                         struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                         struct mca_coll_base_module_2_4_0_t *module);


int ompi_coll_bullnbc_palltoall_init (const void *sbuf, int sparts, int scount, struct ompi_datatype_t *sdtype,
                                      void* rbuf, int rparts, int rcount, struct ompi_datatype_t *rdtype,
                                      struct ompi_communicator_t *comm, struct ompi_info_t *info,
                                      ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_palltoallr_init_select (const void *sbuf, int sparts, int scount, ompi_datatype_t *sdtype,
                                      ompi_request_t ** srequest,
                                      void* rbuf, int rparts, int rcount, ompi_datatype_t *rdtype,
                                      ompi_request_t ** rrequest,
                                      struct ompi_communicator_t *comm, struct ompi_info_t *info,
                                      ompi_request_t ** request,
                                      struct mca_coll_base_module_2_4_0_t *module);
int
ompi_coll_bullnbc_palltoallv_init(const void *sbuf,
                                  const int *sparts, const int *scounts,
                                  const int *sdispls,
                                  struct ompi_datatype_t *sdtype,
                                  void* rbuf,
                                  const int *rparts, const int *rcounts,
                                  const int *rdispls,
                                  struct ompi_datatype_t *rdtype,
                                  struct ompi_communicator_t *comm,
                                  struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int
ompi_coll_bullnbc_palltoallvr_init(const void *sbuf,
                                   const int *sparts, const int *scounts,
                                   const int *sdispls,
                                   struct ompi_datatype_t *sdtype,
                                   ompi_request_t ** srequest,
                                   void* rbuf,
                                   const int *rparts, const int *rcounts,
                                   const int *rdispls,
                                   struct ompi_datatype_t *rdtype,
                                   ompi_request_t ** rrequest,
                                   struct ompi_communicator_t *comm,
                                   struct ompi_info_t *info,
                                   ompi_request_t ** request,
                                   struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_pbcast_init(void *buff, size_t parts, size_t count,
                                  struct ompi_datatype_t *datatype, int root,
                                  struct ompi_communicator_t *comm, struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_pbcastr_init(void *buff, size_t parts, size_t count,
                                  struct ompi_datatype_t *datatype, ompi_request_t ** subreqs, int root,
                                  struct ompi_communicator_t *comm, struct ompi_info_t *info,
                                  ompi_request_t ** request,
                                  struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_pgather_init (const void *sbuf, size_t sendparts, int sendcount,
                                    struct ompi_datatype_t *sdatatype,
                                    void* rbuf, size_t recvparts, int recvcount,
                                    struct ompi_datatype_t *rdatatype,
                                    int root,
                                    struct ompi_communicator_t *comm,
                                    struct ompi_info_t *info,
                                    ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int
ompi_coll_bullnbc_pgatherr_init (const void *sbuf, size_t sendparts, int sendcount,
                                 struct ompi_datatype_t *sdatatype,
                                 ompi_request_t ** sendreqs,
                                 void* rbuf, size_t recvparts, int recvcount,
                                 struct ompi_datatype_t *rdatatype,
                                 ompi_request_t ** recvreqs,
                                 int root,
                                 struct ompi_communicator_t *comm,
                                 struct ompi_info_t *info,
                                 ompi_request_t ** request,
                                 struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_preduce_init (const void *sbuf, void* rbuf,
                                    size_t parts, int count,
                                    struct ompi_datatype_t *datatype,
                                    struct ompi_op_t *op, int root,
                                    struct ompi_communicator_t *comm,
                                    struct ompi_info_t *info,
                                    ompi_request_t ** request,
                                    struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_preducer_init (const void *sbuf, void* rbuf,
                                     size_t parts, int count,
                                     struct ompi_datatype_t *datatype,
                                     ompi_request_t ** sendreqs,
                                     ompi_request_t ** recvreqs,
                                     struct ompi_op_t *op, int root,
                                     struct ompi_communicator_t *comm,
                                     struct ompi_info_t *info,
                                     ompi_request_t ** request,
                                     struct mca_coll_base_module_2_4_0_t *module);

int ompi_coll_bullnbc_neighbor_allgather_init(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                             int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm,
                                             MPI_Info info, ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_neighbor_allgatherv_init(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                              const int *rcounts, const int *displs, MPI_Datatype rtype,
                                              struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                              struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_neighbor_alltoall_init(const void *sbuf, int scount, MPI_Datatype stype, void *rbuf,
                                            int rcount, MPI_Datatype rtype, struct ompi_communicator_t *comm, MPI_Info info,
                                            ompi_request_t ** request, struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_neighbor_alltoallv_init(const void *sbuf, const int *scounts, const int *sdispls, MPI_Datatype stype,
                                             void *rbuf, const int *rcounts, const int *rdispls, MPI_Datatype rtype,
                                             struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                             struct mca_coll_base_module_2_4_0_t *module);
int ompi_coll_bullnbc_neighbor_alltoallw_init(const void *sbuf, const int *scounts, const MPI_Aint *sdisps, struct ompi_datatype_t * const *stypes,
                                             void *rbuf, const int *rcounts, const MPI_Aint *rdisps, struct ompi_datatype_t * const *rtypes,
                                             struct ompi_communicator_t *comm, MPI_Info info, ompi_request_t ** request,
                                             struct mca_coll_base_module_2_4_0_t *module);


END_C_DECLS

#endif /* MCA_COLL_BULLNBC_EXPORT_H */
