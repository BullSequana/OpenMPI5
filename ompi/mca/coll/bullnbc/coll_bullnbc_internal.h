/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006 The Trustees of Indiana University and Indiana
 *                    University Research and Technology
 *                    Corporation.  All rights reserved.
 * Copyright (c) 2006 The Technical University of Chemnitz. All
 *                    rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2014      NVIDIA Corporation.  All rights reserved.
 * Copyright (c) 2015-2018 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 */
#ifndef __NBC_INTERNAL_H__
#define __NBC_INTERNAL_H__
#include "ompi_config.h"

/* correct fortran bindings */
#define NBC_F77_FUNC_ F77_FUNC_

#include "mpi.h"

#include "coll_bullnbc.h"
#include "opal/mca/accelerator/accelerator.h"
#include "ompi/include/ompi/constants.h"
#include "ompi/request/request.h"
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/communicator/communicator.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <assert.h>
#include <math.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dividing very close floats may lead to unexpected roundings */
static inline int
ceil_of_log2 (int val) {
    int ret = 0;
    while (1 << ret < val) {
        ret ++;
    }
    return ret;
}

/* set the number of collectives in nbc.h !!!! */

/* several typedefs for NBC */

/* the function type enum */
typedef enum {
  SEND,
  RECV,
  OP,
  COPY,
  UNPACK
} NBC_Fn_type;

/* the send argument struct */
typedef struct {
  NBC_Fn_type type;
  int count;
  const void *buf;
  MPI_Datatype datatype;
  int dest;
  char tmpbuf;
  bool local;
  bool retain;
} NBC_Args_send;

/* the receive argument struct */
typedef struct {
  NBC_Fn_type type;
  int count;
  void *buf;
  MPI_Datatype datatype;
  char tmpbuf;
  int source;
  bool local;
  bool retain;
} NBC_Args_recv;

/* the operation argument struct */
typedef struct {
  NBC_Fn_type type;
  char tmpbuf1;
  char tmpbuf2;
  const void *buf1;
  void *buf2;
  MPI_Op op;
  MPI_Datatype datatype;
  int count;
  bool retain;
} NBC_Args_op;

/* the copy argument struct */
typedef struct {
  NBC_Fn_type type;
  int srccount;
  const void *src;
  void *tgt;
  MPI_Datatype srctype;
  bool src_retain;
  MPI_Datatype tgttype;
  bool tgt_retain;
  int tgtcount;
  char tmpsrc;
  char tmptgt;
} NBC_Args_copy;

/* unpack operation arguments */
typedef struct {
  NBC_Fn_type type;
  int count;
  void *inbuf;
  void *outbuf;
  MPI_Datatype datatype;
  char tmpinbuf;
  char tmpoutbuf;
} NBC_Args_unpack;

typedef struct {
    NBC_Fn_type type;
    union {
        NBC_Args_send send;
        NBC_Args_recv recv;
        NBC_Args_op op;
        NBC_Args_copy copy;
        NBC_Args_unpack unpack;
    } args;
} BULLNBC_Args_generic;

/* internal function prototypes */

int BULLNBC_Sched_send (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                        int dest, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_send_insert (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                               int dest, BULLNBC_Schedule *schedule, int round, int index);
static inline int
NBC_Sched_send(const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
               int dest, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_send(buf, tmpbuf, count, datatype, dest, schedule, barrier);
}
int BULLNBC_Sched_local_send (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                              int dest, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_local_send_insert (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                                     int dest, BULLNBC_Schedule *schedule, int round, int index);
static inline int
NBC_Sched_local_send (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                      int dest, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_local_send(buf, tmpbuf, count, datatype, dest, schedule, barrier);
}
int BULLNBC_Sched_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                        int source, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_recv_insert (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                               int source, BULLNBC_Schedule *schedule, int round, int index);
static inline int
NBC_Sched_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                int source, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_recv(buf, tmpbuf, count, datatype, source, schedule, barrier);
}
int BULLNBC_Sched_local_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                              int source, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_local_recv_insert (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                                     int source, BULLNBC_Schedule *schedule, int round, int index);
static inline int
NBC_Sched_local_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                      int source, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_local_recv(buf, tmpbuf, count, datatype, source, schedule, barrier);
}
int BULLNBC_Sched_op (const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
                      MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_op_insert (const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
                             MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule,
                             int round, int index);
static inline int
NBC_Sched_op(const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
             MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_op(buf1, tmpbuf1, buf2, tmpbuf2, count, datatype, op, schedule, barrier);
}
int BULLNBC_Sched_copy(const void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt,
                       char tmptgt, int tgtcount, MPI_Datatype tgttype, BULLNBC_Schedule *schedule,
                       bool barrier);
int BULLNBC_Sched_copy_insert (void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt,
                               char tmptgt, int tgtcount, MPI_Datatype tgttype, BULLNBC_Schedule *schedule,
                               int round, int index);
static inline int NBC_Sched_copy(const void *src, char tmpsrc, int srccount, MPI_Datatype srctype,
                                 void *tgt, char tmptgt, int tgtcount, MPI_Datatype tgttype,
                                 BULLNBC_Schedule *schedule, bool barrier)
{
    return BULLNBC_Sched_copy(src, tmpsrc, srccount, srctype, tgt, tmptgt, tgtcount, tgttype, schedule, barrier);
}
int BULLNBC_Sched_unpack (void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype,
                          void *outbuf, char tmpoutbuf, BULLNBC_Schedule *schedule, bool barrier);
int BULLNBC_Sched_unpack_insert (void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype,
                                 void *outbuf, char tmpoutbuf, BULLNBC_Schedule *schedule,
                                 int round, int index);
static inline int
NBC_Sched_unpack(void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype,
                 void *outbuf, char tmpoutbuf, BULLNBC_Schedule *schedule, bool barrier) {
    return BULLNBC_Sched_unpack(inbuf, tmpinbuf, count, datatype, outbuf, tmpoutbuf, schedule, barrier);
}
int BULLNBC_Sched_barrier (BULLNBC_Schedule *schedule);
static inline int
NBC_Sched_barrier(BULLNBC_Schedule *schedule) {
    return BULLNBC_Sched_barrier(schedule);
}
int BULLNBC_Sched_commit (BULLNBC_Schedule *schedule);
static inline int
NBC_Sched_commit (BULLNBC_Schedule *schedule) {
    return BULLNBC_Sched_commit(schedule);
}
int BULLNBC_Start(NBC_Handle *handle);
static inline int
NBC_Start(NBC_Handle *handle) {
    return BULLNBC_Start(handle);
}

void BULLNBC_Return_handle(ompi_coll_bullnbc_request_t *request);
static inline int NBC_Type_intrinsic(MPI_Datatype type);
int NBC_Create_fortran_handle(int *fhandle, NBC_Handle **handle);

int BULLNBC_Schedule_request(BULLNBC_Schedule *schedule, ompi_communicator_t *comm,
                             ompi_coll_bullnbc_module_t *module, bool persistent,
                             ompi_request_t **request, void *tmpbuf);
BULLNBC_Schedule *BULLNBC_create_schedule(int nrounds, int *round_arity, bool dynamic);

/* some macros */

static inline void NBC_Error (char *format, ...) {
  va_list args;

  va_start (args, format);
  char prefixed_fmt[1024];
  snprintf(prefixed_fmt, 1024, "[Bullnbc ERROR] %s\n", format);

  vfprintf (stderr, prefixed_fmt, args);
  fprintf (stderr, "\n");
  fflush(stderr);
  va_end (args);
  ompi_mpi_abort(&ompi_mpi_comm_world.comm, OMPI_ERROR);
}

/*
#define NBC_DEBUG(level, ...) {}
*/

static inline void NBC_DEBUG(int level, const char *fmt, ...)
{
  if(mca_coll_bullnbc_component.debug_verbose >= level) {
    int rank;
    va_list ap;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char prefixed_fmt[1024];
    snprintf(prefixed_fmt, 1024, "[Bullnbc %d] %s\n", rank, fmt);

    va_start(ap, fmt);
    vfprintf(stderr, prefixed_fmt, ap);
    va_end (ap);
    fflush(stderr);
  }
}

/* returns true (1) or false (0) if type is intrinsic or not */
static inline int NBC_Type_intrinsic(MPI_Datatype type) {

  if( ( type == MPI_INT ) ||
      ( type == MPI_LONG ) ||
      ( type == MPI_SHORT ) ||
      ( type == MPI_UNSIGNED ) ||
      ( type == MPI_UNSIGNED_SHORT ) ||
      ( type == MPI_UNSIGNED_LONG ) ||
      ( type == MPI_FLOAT ) ||
      ( type == MPI_DOUBLE ) ||
      ( type == MPI_LONG_DOUBLE ) ||
      ( type == MPI_BYTE ) ||
      ( type == MPI_FLOAT_INT) ||
      ( type == MPI_DOUBLE_INT) ||
      ( type == MPI_LONG_INT) ||
      ( type == MPI_2INT) ||
      ( type == MPI_SHORT_INT) ||
      ( type == MPI_LONG_DOUBLE_INT))
    return 1;
  else
    return 0;
}

/* let's give a try to inline functions */
static inline int NBC_Copy(const void *src, int srccount, MPI_Datatype srctype, void *tgt, int tgtcount, MPI_Datatype tgttype, MPI_Comm comm) {
  int res;

  res = ompi_datatype_sndrcv(src, srccount, srctype, tgt, tgtcount, tgttype);
  if (OMPI_SUCCESS != res) {
    NBC_Error ("MPI Error in ompi_datatype_sndrcv() (%i)", res);
    return res;
  }

  return OMPI_SUCCESS;
}

static inline int NBC_Unpack(const void *src, int srccount, MPI_Datatype srctype, void *tgt, MPI_Comm comm) {
  size_t size;
  int res;
  ptrdiff_t ext, lb;
  uint64_t flags;
  int is_accel_buf1, is_accel_buf2;
  int dev_id;

  is_accel_buf1 = opal_accelerator.check_addr(tgt, &dev_id, &flags);
  is_accel_buf2 = opal_accelerator.check_addr(src, &dev_id, &flags);
  if (is_accel_buf1 < 0) {
    return is_accel_buf1;
  } else if (is_accel_buf2 < 0) {
    return is_accel_buf2;
  }

  if(NBC_Type_intrinsic(srctype) &&
     is_accel_buf1 == 0 &&
     is_accel_buf2 == 0) {
    /* if we have the same types and they are contiguous (intrinsic
     * types are contiguous), we can just use a single memcpy */
    res = ompi_datatype_get_extent (srctype, &lb, &ext);
    if (OMPI_SUCCESS != res) {
      NBC_Error ("MPI Error in MPI_Type_extent() (%i)", res);
      return res;
    }

    memcpy(tgt, src, srccount * ext);

  } else {
    /* we have to unpack */
    res = ompi_datatype_type_size(srctype, &size);
    if (MPI_SUCCESS != res) {
      NBC_Error ("MPI Error in ompi_datatype_type_size() (%i)", res);
      return res;
    }
    size *= srccount;
    res = ompi_datatype_sndrcv(src, (int32_t)size, MPI_BYTE, tgt, srccount, srctype);
    if (MPI_SUCCESS != res) {
      NBC_Error ("MPI Error in data unpacking (%i)", res);
      return res;
    }
  }

  return OMPI_SUCCESS;
}

static inline int
bullnbc_get_noop_request(bool persistent, ompi_request_t **request) {
  if (persistent) {
    return ompi_request_persistent_noop_create(request);
  } else {
    *request = &ompi_request_empty;
    return OMPI_SUCCESS;
  }
}
#define NBC_IN_PLACE(sendbuf, recvbuf, inplace) \
{ \
  inplace = 0; \
  if(recvbuf == sendbuf) { \
    inplace = 1; \
  } else \
  if(sendbuf == MPI_IN_PLACE) { \
    sendbuf = recvbuf; \
    inplace = 1; \
  } else \
  if(recvbuf == MPI_IN_PLACE) { \
    recvbuf = (void *)sendbuf; \
    inplace = 1; \
  } \
}

int BULLNBC_Comm_neighbors_count (ompi_communicator_t *comm, int *indegree, int *outdegree);
int BULLNBC_Comm_neighbors (ompi_communicator_t *comm, int **sources, int *source_count, int **destinations, int *dest_count);

static inline int
module_need_progress(ompi_coll_bullnbc_module_t *module)
{
    int32_t is_registered = opal_atomic_swap_32(&module->comm_registered, 1);

    /* register progress */
    if (!is_registered) {
        int32_t tmp;
        tmp = OPAL_THREAD_ADD_FETCH32(&mca_coll_bullnbc_component.active_comms, 1);
        if (1 == tmp) {
            opal_progress_register(ompi_coll_bullnbc_progress);
        }
    }
    return OMPI_SUCCESS;
}



static inline void *
bullnbc_xmalloc_internal (size_t size, const char * func, int line) {
    if (0 == size) {
        return NULL;
    }

    void * ptr = malloc(size);
    if (NULL == ptr) {
        NBC_Error("Failed to malloc (%s,%d)\n", func, line);
    }
    return ptr;
}
#define bullnbc_xmalloc(_size) bullnbc_xmalloc_internal(_size, __func__, __LINE__)

#ifdef __cplusplus
}
#endif

#endif
