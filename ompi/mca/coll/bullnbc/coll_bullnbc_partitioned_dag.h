/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#ifndef MCA_COLL_BULLNBC_PARTITIONED_DAG_H
#define MCA_COLL_BULLNBC_PARTITIONED_DAG_H

BEGIN_C_DECLS

#include "ompi/mpiext/pcoll_part/c/mpiext_pcoll_part_c.h"
/*
 * @file
 *
 * This file describes objects and primitives to build a DAG.
 * Each dag item have a type that define what operation needs to be
 * performed when the node is unlokced at runtime.
 * Items contains the data for this action.
 * Once the action is complete, dependencies are unlocked.
 */

#define PROFILE 0

/* struct forward declaration & function prototypes */
typedef struct ompi_coll_bullnbc_pcoll_request_t ompi_coll_bullnbc_pcoll_request_t;


typedef enum {
  DAG_CONVERT,
  DAG_COPY,
  DAG_OP,
  DAG_SEND,
  DAG_RECV,

#if OMPI_MPI_NOTIFICATIONS
  DAG_PUT,
  DAG_NOTIF,
#endif

  DAG_READY,
  DAG_COMPLETE,
  N_DAG_TYPE,
} pcoll_dag_item_type;

static inline const char * dag_item_type_to_str(pcoll_dag_item_type type){
    switch(type){
        case DAG_CONVERT:
            return "Conv";
        case DAG_COPY:
            return "Cpy";
        case DAG_OP:
            return "Op";
        case DAG_SEND:
            return "Send";
        case DAG_RECV:
            return "Rcv";
#if OMPI_MPI_NOTIFICATIONS
        case DAG_PUT:
            return "Put";
        case DAG_NOTIF:
            return "Wt notif";
#endif /* OMPI_MPI_NOTIFICATIONS */
        case DAG_READY:
            return "Rdy";
        case DAG_COMPLETE:
            return "Comp";
        default:
            abort();
    }
}

/* the send argument struct */
typedef struct {
    const void *buf;
    size_t count;
    struct ompi_datatype_t *datatype;
    unsigned int peer;
    struct ompi_communicator_t* comm;
    unsigned int tagshift;

    ompi_coll_bullnbc_pcoll_request_t * collreq;
    ompi_request_t ** req;
    unsigned int started;
} bullnbc_comm_t;

/* the copy argument struct */
typedef struct {
  char *dst;
  char *src;
  size_t size;
} bullnbc_copy_t;

/* pack/unpack operation arguments */
typedef struct {
  const void *inbuf;
  void *outbuf;
  unsigned int in_count;
  unsigned int out_count;
  ompi_datatype_t* in_ddt;
  ompi_datatype_t* out_ddt;
} bullnbc_convert_t;

typedef struct {
    ompi_coll_bullnbc_pcoll_request_t * collreq;
    ompi_request_t * req;
} bullnbc_complete_t;

typedef struct {
  char *dst;
  const char *src;
  unsigned int size;
  ompi_datatype_t* ddt;
  MPI_Op op;
} bullnbc_op_t;

#if OMPI_MPI_NOTIFICATIONS
typedef struct {
    const void *buf;
    size_t scount;
    struct ompi_datatype_t *sddt;
    unsigned int dst;
    ptrdiff_t disp;
    size_t rcount;
    struct ompi_datatype_t *rddt;
    ompi_win_t* win;
    unsigned int notif_id;
} bullnbc_put_t;

typedef struct {
    ompi_win_t* win;
    unsigned int id;
} bullnbc_wait_notif_t;
#endif



struct pcoll_dag_item;
struct pcoll_dag_item {
    opal_free_list_item_t super;

    /* Partitioned Pt2pt cannot start before completion of all partitions
     * (cf MPI4 section 4.2.2). Therefore partitioned collective cannot call
     * start before completion, as they may depend on partitioned pt2pt.
     * So counts of waited previous steps can be stored directly in the DAG. */
    int32_t backward_deps;
    int32_t current_unlock;

    pcoll_dag_item_type type;
    union {
        bullnbc_comm_t comm;
        bullnbc_copy_t copy;
        bullnbc_convert_t convert;
        bullnbc_op_t op;
        bullnbc_complete_t complete;
#if OMPI_MPI_NOTIFICATIONS
        bullnbc_put_t put;
        bullnbc_wait_notif_t notif;
#endif /* OMPI_MPI_NOTIFICATIONS */
    } args;

    unsigned int n_started; /* Debug */
    unsigned int max_next_steps;
    unsigned int n_next_steps;
    struct pcoll_dag_item* next_step; /* If count =1 */
    /* Same object can be in deps of multiple item: lists cannot be used */
    struct pcoll_dag_item** next_steps; /* If count > 1 */
};
typedef struct pcoll_dag_item pcoll_dag_item;
OBJ_CLASS_DECLARATION(pcoll_dag_item);



ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_alloc_pcoll_request_dag (void);
int dag_alloc_user_subreqs(ompi_coll_bullnbc_pcoll_request_t *req,
                           size_t total_sparts,
                           ompi_request_t **user_sreqs,
                           size_t total_rparts,
                           ompi_request_t **user_rreqs);
int dag_free_user_subreqs(ompi_coll_bullnbc_pcoll_request_t *req);

ompi_coll_bullnbc_pcoll_request_t*
ompi_mca_coll_bullnbc_pcoll_init_dag_req(size_t total_sparts,
                                         ompi_request_t** user_sreqs,
                                         size_t total_rparts,
                                         ompi_request_t** user_rreqs,
                                         unsigned int n_internal_reqs,
                                         unsigned int n_recv,
                                         struct ompi_communicator_t *comm,
                                         ompi_request_t ** user_main_req);
ompi_coll_bullnbc_pcoll_request_t*
coll_bullnbc_pcoll_init_dag_ntag_req(size_t total_sparts,
                                     ompi_request_t** user_sreqs,
                                     size_t total_rparts,
                                     ompi_request_t** user_rreqs,
                                     unsigned int n_internal_reqs,
                                     unsigned int n_recv,
                                     struct ompi_communicator_t *comm,
                                     int ntags,
                                     ompi_request_t ** user_main_req);
void coll_bullnbc_fini_part_dag(void);
void coll_bullnbc_init_part_dag(void);

void run_on_start(pcoll_dag_item *item, ompi_coll_bullnbc_pcoll_request_t *req);
void create_dependency(pcoll_dag_item* item, pcoll_dag_item* deps);
pcoll_dag_item* sched_ready(unsigned int ready_idx, ompi_coll_bullnbc_pcoll_request_t* req, unsigned int next_deps);
pcoll_dag_item* sched_complete(unsigned int complete_idx, ompi_coll_bullnbc_pcoll_request_t* req);
pcoll_dag_item* sched_convertion(const void * from_buf, unsigned int from_count, ompi_datatype_t *from_ddt, void* to_buf, unsigned int to_count, ompi_datatype_t* to_ddt, unsigned int next_deps);
pcoll_dag_item* sched_op(void * dst, const void* src, unsigned int count, ompi_datatype_t* ddt, MPI_Op mpi_op, unsigned int next_deps);

pcoll_dag_item* sched_send(const void * buf, unsigned int count, ompi_datatype_t *ddt, unsigned int dst, struct ompi_communicator_t* comm, ompi_request_t ** isend_req, ompi_coll_bullnbc_pcoll_request_t * req, unsigned int next_deps);
pcoll_dag_item* sched_send_tagged(const void * buf, unsigned int count, ompi_datatype_t *ddt, unsigned int dst, unsigned int tagshift, struct ompi_communicator_t* comm, ompi_request_t ** isend_req, ompi_coll_bullnbc_pcoll_request_t * req, unsigned int next_deps);

#define sched_send_tagged_v2(buf, count, ddt, dst, tag, comm, req) \
    sched_send_tagged(buf, count, ddt, dst, tag, comm, NULL, req, 0)
#define sched_send_v2(buf, count, ddt, dst, comm, req) \
    sched_send_tagged(buf, count, ddt, dst, 0, comm, NULL, req, 0)

#define sched_recv_tagged_v2(buf, count, ddt, src, tag, comm, req) \
    sched_recv_tagged(buf, count, ddt, src, tag, comm, NULL ,req, -1, 0)
#define sched_recv_v2(buf, count, ddt, src, comm, req) \
    sched_recv_tagged(buf, count, ddt, src, 0, comm, NULL ,req, -1, 0)
pcoll_dag_item* sched_recv(void * buf, unsigned int count, ompi_datatype_t *ddt, unsigned int src, struct ompi_communicator_t* comm, ompi_request_t ** irecv_req, ompi_coll_bullnbc_pcoll_request_t * req, unsigned int dag_entry_idx, unsigned int next_deps);
pcoll_dag_item* sched_recv_tagged(void * buf, unsigned int count, ompi_datatype_t *ddt, unsigned int src, unsigned int tagshift, struct ompi_communicator_t* comm, ompi_request_t ** irecv_req, ompi_coll_bullnbc_pcoll_request_t * req, unsigned int dag_entry_idx, unsigned int next_deps);

#if OMPI_MPI_NOTIFICATIONS
pcoll_dag_item* sched_put(void * buf, unsigned int scount, ompi_datatype_t *sddt, unsigned int dst, ptrdiff_t disp, unsigned int rcount, ompi_datatype_t* rddt, ompi_coll_bullnbc_pcoll_request_t * collreq, unsigned int notif, unsigned int next_deps);
pcoll_dag_item* sched_notif(ompi_coll_bullnbc_pcoll_request_t* collreq, unsigned int notif_id, unsigned int dag_entry_idx,unsigned int next_deps);
#endif /* OMPI_MPI_NOTIFICATIONS */

END_C_DECLS

#endif /* MCA_COLL_BULLNBC_PARTITIONED_DAG_H */
