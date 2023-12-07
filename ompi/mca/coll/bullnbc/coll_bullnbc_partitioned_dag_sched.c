/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "coll_bullnbc_internal.h"

#include "coll_bullnbc_partitioned.h"
#include "coll_bullnbc_partitioned_dag.h"
#if OMPI_MPI_NOTIFICATIONS
#include "ompi/runtime/params.h"
#endif /* OMPI_MPI_NOTIFICATIONS */

/* Item is triggered on request start */
void
run_on_start(pcoll_dag_item *item, ompi_coll_bullnbc_pcoll_request_t *req) {
    int dag_entry_idx = req->total_sparts + req->n_entry_sched;
    req->n_entry_sched ++;
    if (dag_entry_idx >= req->n_entry_nodes) {
        NBC_Error("Try to sched on start item %p #%d, %d avail",
                  item, dag_entry_idx, req->n_entry_nodes);
    }
    req->entry_nodes[dag_entry_idx] = item;
    NBC_DEBUG(5, "Sched %p (type %d) on request %p start\n",
            item, item->type, req);
    return;
}


/*
 * @file
 *
 * This files provides routines to schedule DAG tasks.
 * This makes DAG collectives implementations simplier and shorter.
 * The focus can then be put on the algoithms.
 */

void
create_dependency(pcoll_dag_item* item, pcoll_dag_item* deps)
{
    NBC_DEBUG(5, "link %s %p --%d--> %s %p\n",
              dag_item_type_to_str(item->type), item,
              item->n_next_steps,
              dag_item_type_to_str(deps->type), deps);
    deps->backward_deps++;
    deps->current_unlock++;

    if (OPAL_UNLIKELY(item->max_next_steps <= item->n_next_steps)) {
        NBC_DEBUG(100, "WARNING sched %s %p initialy planned %d deps, needs %d to append%s %p\n",
                  dag_item_type_to_str(item->type), item,
                  item->max_next_steps,
                  item->n_next_steps +1,
                  dag_item_type_to_str(deps->type), deps);
        if (0 == item->max_next_steps) {
            item->max_next_steps = 1;
        } else {
            item->max_next_steps *= 2;
        }
        size_t size = item->max_next_steps * sizeof(pcoll_dag_item*);
        void * tmp = realloc (item->next_steps, size);
        if (NULL == tmp) {
            NBC_Error("Failed to realloc deps\n");
            abort();
        }
        item->next_steps = tmp;
    }
    item->next_steps[item->n_next_steps] = deps;
    item->n_next_steps++;
}

static inline pcoll_dag_item*
new_dag_item(unsigned int next_deps)
{
    pcoll_dag_item* item = malloc(sizeof(pcoll_dag_item));
    item->n_started = 0;
    item->backward_deps = 0;
    item->current_unlock = 0;

    item->n_next_steps = 0;
    item->max_next_steps = next_deps;
    item->next_steps = malloc(next_deps * sizeof(pcoll_dag_item*));
    return item;
}

pcoll_dag_item*
sched_ready(unsigned int ready_idx, ompi_coll_bullnbc_pcoll_request_t* req, unsigned int next_deps)
{
    pcoll_dag_item* ready = new_dag_item(next_deps);
    ready->type = DAG_READY;
    req->entry_nodes[ready_idx] = ready;
    return ready;
}

pcoll_dag_item*
sched_complete(unsigned int complete_idx, ompi_coll_bullnbc_pcoll_request_t* req)
{
    pcoll_dag_item* complete = new_dag_item(0);
    complete->type = DAG_COMPLETE;
    complete->args.complete.req = req->user_part_reqs[complete_idx];
    complete->args.complete.collreq = req;
    return complete;
}


pcoll_dag_item*
sched_convertion(const void * from_buf, unsigned int from_count, ompi_datatype_t *from_ddt,
               void* to_buf, unsigned int to_count, ompi_datatype_t* to_ddt,
               unsigned int next_deps)
{
    pcoll_dag_item* convert = new_dag_item(next_deps);
    NBC_DEBUG(5, "Sched conv %p: %p -> %p\n", convert, from_buf,to_buf);
    convert->type = DAG_CONVERT;
    convert->args.convert.inbuf = from_buf;
    convert->args.convert.in_count = from_count;
    convert->args.convert.in_ddt = from_ddt;
    if (!ompi_datatype_is_predefined(from_ddt)){
        OBJ_RETAIN(from_ddt);
    }
    convert->args.convert.outbuf = to_buf;
    convert->args.convert.out_count = to_count;
    convert->args.convert.out_ddt = to_ddt;
    if (!ompi_datatype_is_predefined(to_ddt)){
        OBJ_RETAIN(to_ddt);
    }
    return convert;
}

pcoll_dag_item*
sched_op(void * dst, const void* src, unsigned int count, ompi_datatype_t* ddt, MPI_Op mpi_op,
         unsigned int next_deps)
{
    pcoll_dag_item* op = new_dag_item(next_deps);
    NBC_DEBUG(5, "Sched OP %p from %p to %p with %d ddt\n",
              op, src, dst,  count);
    op->type=DAG_OP;
    op->args.op.dst = dst;
    op->args.op.src = src;
    op->args.op.size = count;
    op->args.op.ddt = ddt;
    if (!ompi_datatype_is_predefined(ddt)){
        OBJ_RETAIN(ddt);
    }
    op->args.op.op = mpi_op;

    return op;
}


pcoll_dag_item*
sched_send_tagged(const void * buf, unsigned int count, ompi_datatype_t *ddt,
           unsigned int dst, unsigned int tagshift, struct ompi_communicator_t* comm,
           ompi_request_t ** isend_req,
           ompi_coll_bullnbc_pcoll_request_t * req,
           unsigned int next_deps)
{
    pcoll_dag_item* send = new_dag_item(next_deps);
    NBC_DEBUG(5, "Sched send %p from %p to %d with %d ddt %p, req=%p\n",
            send, buf, dst, count, ddt, isend_req);
    send->type = DAG_SEND;
    send->args.comm.buf = buf;
    send->args.comm.count = count;
    send->args.comm.datatype = ddt;
    send->args.comm.peer = dst;
    send->args.comm.tagshift = tagshift;
    send->args.comm.comm = comm;
    if (NULL == isend_req) {
        isend_req = &req->internal_reqs[req->n_internal_reqs_sched];
    }
    req->n_internal_reqs_sched++;
    if (req->n_internal_reqs_sched > req->n_internal_reqs) {
        NBC_Error("Too many internal comms wih this send(%d/%d)\n",
                  req->n_internal_reqs_sched , req->n_internal_reqs);
        abort();
    }
    send->args.comm.req = isend_req;
    send->args.comm.collreq = req;
    send->args.comm.started = 0;

    return send;
}

pcoll_dag_item*
sched_send(const void * buf, unsigned int count, ompi_datatype_t *ddt,
           unsigned int dst, struct ompi_communicator_t* comm,
           ompi_request_t ** isend_req,
           ompi_coll_bullnbc_pcoll_request_t * req,
           unsigned int next_deps)
{
    return sched_send_tagged(buf, count, ddt, dst, 0, comm,
                             isend_req, req, next_deps);
}


pcoll_dag_item*
sched_recv_tagged(void * buf, unsigned int count, ompi_datatype_t *ddt,
           unsigned int src, unsigned int tagshift,
           struct ompi_communicator_t* comm,
           ompi_request_t ** irecv_req,
           ompi_coll_bullnbc_pcoll_request_t * req,
           unsigned int dag_entry_idx, unsigned int next_deps)
{
    pcoll_dag_item* recv = new_dag_item(next_deps);
    recv->type = DAG_RECV;
    recv->args.comm.buf = buf;
    recv->args.comm.count = count;
    recv->args.comm.datatype = ddt;
    recv->args.comm.peer = src;
    recv->args.comm.tagshift = tagshift;
    recv->args.comm.comm = comm;
    if (NULL == irecv_req) {
        irecv_req = &req->internal_reqs[req->n_internal_reqs_sched];
    }
    req->n_internal_reqs_sched++;
    if (req->n_internal_reqs_sched > req->n_internal_reqs) {
        NBC_Error("Too many internal comms wih this recv(%d/%d)\n",
                  req->n_internal_reqs_sched , req->n_internal_reqs);
        abort();
    }
    recv->args.comm.req = irecv_req;
    recv->args.comm.collreq = req;
    if (-1 == dag_entry_idx) {
        dag_entry_idx = req->total_sparts + req->n_entry_sched;
        if (dag_entry_idx >= req->n_entry_nodes){
            abort();
        }
    }
    req->n_entry_sched++;
    req->entry_nodes[dag_entry_idx] = recv;

    NBC_DEBUG(5, "Sched recv#%d %p from peer %d to addr %p with %d ddt %p, pml_req=%p\n",
            dag_entry_idx, recv,
            recv->args.comm.peer,
            recv->args.comm.buf,
            recv->args.comm.count,
            recv->args.comm.datatype,
            recv->args.comm.req);
    return recv;
}

pcoll_dag_item*
sched_recv(void * buf, unsigned int count, ompi_datatype_t *ddt,
           unsigned int src, struct ompi_communicator_t* comm,
           ompi_request_t ** irecv_req,
           ompi_coll_bullnbc_pcoll_request_t * req,
           unsigned int dag_entry_idx, unsigned int next_deps)
{
    return sched_recv_tagged(buf, count, ddt, src, 0, comm, irecv_req,
                             req, dag_entry_idx, next_deps);
}
#if OMPI_MPI_NOTIFICATIONS
pcoll_dag_item*
sched_put(void * buf, unsigned int scount, ompi_datatype_t *sddt, unsigned int dst,
          ptrdiff_t disp, unsigned int rcount, ompi_datatype_t* rddt,
          ompi_coll_bullnbc_pcoll_request_t * collreq,
          unsigned int notif,
          unsigned int next_deps)
{
    if (notif > ompi_max_notification_idx) {
        NBC_Error("Notification %d is too high, max is %d",
                  notif, ompi_max_notification_idx);
        abort();
    }
    pcoll_dag_item* put = new_dag_item(next_deps);
    put->type = DAG_PUT;
    put->args.put.buf = buf;
    put->args.put.scount = scount;
    put->args.put.sddt = sddt;
    put->args.put.dst = dst;
    put->args.put.disp = disp;
    put->args.put.rcount = rcount;
    put->args.put.rddt = rddt;
    put->args.put.win = collreq->win;
    put->args.put.notif_id = notif;
    NBC_DEBUG(5, "Sched put %d of %d elems at offset +%d on win %p to peer %d\n",
              notif, rcount, disp, collreq->win, dst);

    return put;
}

pcoll_dag_item*
sched_notif(ompi_coll_bullnbc_pcoll_request_t* collreq,
            unsigned int notif_id,
            unsigned int dag_entry_idx, unsigned int next_deps)
{
    pcoll_dag_item* notif = new_dag_item(next_deps);
    notif->type = DAG_NOTIF;
    notif->args.notif.win= collreq->win;
    notif->args.notif.id= notif_id;
    collreq->n_entry_sched++;
    collreq->entry_nodes[dag_entry_idx] = notif;
    NBC_DEBUG(5, "Sched notif %p on win %p\n", notif, collreq->win);

    return notif;
}
#endif /* OMPI_MPI_NOTIFICATIONS */
