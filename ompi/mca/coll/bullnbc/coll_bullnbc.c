/* -*- Mode: C; c-basic-offset:2 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2006      The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2013-2018 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2006      The Technical University of Chemnitz. All
 *                         rights reserved.
 * Copyright (c) 2015      Los Alamos National Security, LLC.  All rights
 *                         reserved.
 * Copyright (c) 2015-2019 Research Organization for Information Science
 *                         and Technology (RIST).  All rights reserved.
 *
 * Author(s): Torsten Hoefler <htor@cs.indiana.edu>
 *
 * Copyright (c) 2012      Oracle and/or its affiliates.  All rights reserved.
 * Copyright (c) 2016      IBM Corporation.  All rights reserved.
 * Copyright (c) 2017      Ian Bradley Morgan and Anthony Skjellum. All
 *                         rights reserved.
 * Copyright (c) 2018      FUJITSU LIMITED.  All rights reserved.
 * Copyright (c) 2021-2023 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 */
#include "coll_bullnbc_internal.h"
#include "ompi/mca/coll/base/coll_base_util.h"
#include "ompi/op/op.h"
#include "ompi/mca/pml/pml.h"

/* only used in this file */
static inline int BULLNBC_Start_round(NBC_Handle *handle);
static inline int BULLNBC_append_round_cmd(BULLNBC_Schedule *schedule, int round,
                                           int index, BULLNBC_Args_generic *cmd);
static inline int BULLNBC_Sched_internal_push(BULLNBC_Schedule *schedule, bool barrier, int *round, int *arity);

#define NBC_SCHEDULE_MIN_DATA_SIZE 4096
#define BULLNBC_DEFAULT_MIN_CMD     1024
#define BULLNBC_DEFAULT_MIN_ROUND   1024

static void nbc_schedule_constructor (BULLNBC_Schedule *schedule) {
  schedule->dynamic = true;
  schedule->max_rounds = BULLNBC_DEFAULT_MIN_ROUND;
  schedule->max_cmds = BULLNBC_DEFAULT_MIN_CMD;
  schedule->num_rounds = 0;
  schedule->num_cmds = 0;
  schedule->max_round_arity = 0;

  schedule->rounds = NULL;      /* allocate in commit */
  schedule->requests = NULL;    /* allocate in commit */

  schedule->cmds = malloc(sizeof(BULLNBC_Args_generic) * schedule->max_cmds);
  schedule->arity = malloc(sizeof(int) * schedule->max_rounds);
  memset(schedule->arity, 0, sizeof(int) * schedule->max_rounds);
}

static void nbc_schedule_destructor (BULLNBC_Schedule *schedule) {
    free(schedule->cmds);
    schedule->cmds = NULL;
    free(schedule->arity);
    schedule->arity = NULL;
    free(schedule->rounds);
    schedule->rounds = NULL;
    free(schedule->requests);
    schedule->requests = NULL;
}

OBJ_CLASS_INSTANCE(BULLNBC_Schedule, opal_free_list_item_t, nbc_schedule_constructor,
                   nbc_schedule_destructor);



/* this function puts an operation into the schedule */
static int
BULLNBC_Sched_op_internal (const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
                               MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule,
                               int round, int index) {
  int ret;
  BULLNBC_Args_generic gen_args;;

  /* store the passed arguments */
  gen_args.type = OP;
  gen_args.args.op.type = OP;
  gen_args.args.op.buf1 = buf1;
  gen_args.args.op.buf2 = buf2;
  gen_args.args.op.tmpbuf1 = tmpbuf1;
  gen_args.args.op.tmpbuf2 = tmpbuf2;
  gen_args.args.op.count = count;
  gen_args.args.op.op = op;
  gen_args.args.op.datatype = datatype;
  /* some algorithm create their own datatype */
  if (NULL != datatype && !ompi_datatype_is_predefined(datatype)) {
    gen_args.args.op.retain = true;
    OBJ_RETAIN(gen_args.args.op.datatype);
  } else {
      gen_args.args.op.retain = false;
  }

  /* append to the round-schedule */
  ret = BULLNBC_append_round_cmd(schedule, round, index, &gen_args);
  if (OMPI_SUCCESS != ret) {
    return ret;
  }


  NBC_DEBUG(10, "added op2 - ends at round: %d idx: %d\n", round, index);;

  return OMPI_SUCCESS;
}

int BULLNBC_Sched_op(const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
                     MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule, bool barrier)
{
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_op_internal(buf1, tmpbuf1, buf2, tmpbuf2, count, datatype, op,
                                     schedule, round, index);
}

#if 0
__opal_attribute_unused__
int BULLNBC_Sched_op_insert(const void* buf1, char tmpbuf1, void* buf2, char tmpbuf2, int count,
                            MPI_Datatype datatype, MPI_Op op, BULLNBC_Schedule *schedule,
                            int round, int index)
{
    //TODO: check dynamic !? 
    return BULLNBC_Sched_op_internal(buf1, tmpbuf1, buf2, tmpbuf2, count, datatype, op,
                                     schedule, round, index);
}
#endif

/* this function puts a copy into the schedule */
static inline int BULLNBC_Sched_copy_internal(const void *src, char tmpsrc, int srccount,
                                              MPI_Datatype srctype, void *tgt, char tmptgt,
                                              int tgtcount, MPI_Datatype tgttype,
                                              BULLNBC_Schedule *schedule, int round, int index)
{
    int ret;
    BULLNBC_Args_generic gen_args;
  
    /* store the passed arguments */
    gen_args.type = COPY;
    gen_args.args.copy.type = COPY;
    gen_args.args.copy.src = src;
    gen_args.args.copy.tmpsrc = tmpsrc;
    gen_args.args.copy.srccount = srccount;
    gen_args.args.copy.srctype = srctype;
    if (NULL != srctype && !ompi_datatype_is_predefined(srctype)) {
        gen_args.args.copy.src_retain = true;
        OBJ_RETAIN(gen_args.args.copy.srctype);
    } else {
        gen_args.args.copy.src_retain = false;
    }
    gen_args.args.copy.tgt = tgt;
    gen_args.args.copy.tmptgt = tmptgt;
    gen_args.args.copy.tgtcount = tgtcount;
    gen_args.args.copy.tgttype = tgttype;
    if (NULL != tgttype && !ompi_datatype_is_predefined(tgttype)) {
        gen_args.args.copy.tgt_retain = true;
        OBJ_RETAIN(gen_args.args.copy.tgttype);
    } else {
        gen_args.args.copy.tgt_retain = false;
    }

    /* append to the round-schedule */
    ret =  BULLNBC_append_round_cmd(schedule, round, index, &gen_args); 
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    NBC_DEBUG(10, "added copy - ends at round: %d index: %d\n", round, index);
    return OMPI_SUCCESS;
}

int BULLNBC_Sched_copy(const void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt,
                       char tmptgt, int tgtcount, MPI_Datatype tgttype, BULLNBC_Schedule *schedule,
                       bool barrier)
{
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_copy_internal(src, tmpsrc, srccount, srctype, tgt, tmptgt,
                                       tgtcount, tgttype, schedule, round, index);
}

#if 0
__opal_attribute_unused__
int BULLNBC_Sched_copy_insert(void *src, char tmpsrc, int srccount, MPI_Datatype srctype, void *tgt, char tmptgt,
                              int tgtcount, MPI_Datatype tgttype, BULLNBC_Schedule *schedule, int round, int index)
{
    return BULLNBC_Sched_copy_internal(src, tmpsrc, srccount, srctype, tgt, tmptgt,
                                       tgtcount, tgttype, schedule, round, index);
}
#endif

/* this function puts a unpack into the schedule */
static inline int 
BULLNBC_Sched_unpack_internal (void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype, void *outbuf,
                               char tmpoutbuf, BULLNBC_Schedule *schedule, int round, int index) {
  int ret;
  BULLNBC_Args_generic gen_args;

  /* store the passed arguments */
  gen_args.type = UNPACK;
  gen_args.args.unpack.type = UNPACK;
  gen_args.args.unpack.inbuf = inbuf;
  gen_args.args.unpack.tmpinbuf = tmpinbuf;
  gen_args.args.unpack.count = count;
  gen_args.args.unpack.datatype = datatype;
  gen_args.args.unpack.outbuf = outbuf;
  gen_args.args.unpack.tmpoutbuf = tmpoutbuf;

  /* append to the round-schedule */
  ret = BULLNBC_append_round_cmd(schedule, round, index, &gen_args);
  if (OMPI_SUCCESS != ret) {
    return ret;
  }

  NBC_DEBUG(10, "added send - ends at round: %d idx: %d\n", round, index);

  return OMPI_SUCCESS;
}

int BULLNBC_Sched_unpack(void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype, void *outbuf,
                         char tmpoutbuf, BULLNBC_Schedule *schedule, bool barrier) {
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_unpack_internal(inbuf, tmpinbuf, count, datatype, outbuf,
                                         tmpoutbuf, schedule, round, index); 
}

#if 0
__opal_attribute_unused__
int BULLNBC_Sched_unpack_insert(void *inbuf, char tmpinbuf, int count, MPI_Datatype datatype, void *outbuf,
                                char tmpoutbuf, BULLNBC_Schedule *schedule, int round, int index) {
    return BULLNBC_Sched_unpack_internal(inbuf, tmpinbuf, count, datatype, outbuf,
                                         tmpoutbuf, schedule, round, index);
}
#endif

void BULLNBC_Return_handle(ompi_coll_bullnbc_request_t *request) {
  //NBC_Free (request);
  OMPI_COLL_BULLNBC_REQUEST_RETURN(request);
}

int  BULLNBC_Init_comm(MPI_Comm comm, ompi_coll_bullnbc_module_t *module) {
  return OMPI_SUCCESS;
}

static inline void BULLNBC_release_schedule(BULLNBC_Schedule *schedule) {
    nbc_schedule_destructor(schedule);
    opal_free_list_return(&mca_coll_bullnbc_component.schedules,
                          (opal_free_list_item_t*)schedule);
}

static inline void
BULLNBC_Free (NBC_Handle* handle) {
  if (NULL != handle->schedule) {
    BULLNBC_Schedule *schedule = (BULLNBC_Schedule *) handle->schedule;
    /* release schedule */
    BULLNBC_release_schedule(schedule);
    handle->schedule = NULL;
    handle->req_array = NULL;
  }

  /* if the nbc_I<collective> attached some data */
  /* problems with schedule cache here, see comment (TODO) in
   * nbc_internal.h */
  if (NULL != handle->tmpbuf) {
    free((void*)handle->tmpbuf);
    handle->tmpbuf = NULL;
  }
}

static inline int
BULLNBC_append_round_cmd(BULLNBC_Schedule *schedule, int round, int index, BULLNBC_Args_generic *cmd) {
    BULLNBC_Args_generic *cmds;

    if (round >= schedule->num_rounds) {
        NBC_Error("Invalid arguments round index (%d > %d)",
                  round, schedule->num_rounds);
        abort();    /* should be remove */
    }

    if (index >= schedule->arity[round]) {
        NBC_Error("Invalid arguments round local index (%d > %d)",
                  index, schedule->arity[round]);
        abort();    /* should be remove */
    }

    int idx = 0;

    if (!schedule->dynamic) { 
        /* Get cmd offset */
        for (int i = 0; i < round; i++) {
            idx += schedule->arity[i];
        }

        idx += index;
    } else {
        idx = schedule->num_cmds++;
    }

    if (schedule->dynamic && idx >= schedule->max_cmds) {
        void *tmp;

        schedule->max_cmds *= 2;
        tmp = realloc(schedule->cmds, sizeof(BULLNBC_Args_generic) * schedule->max_cmds);
        if (NULL == tmp) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }

        schedule->cmds = tmp;
    }
    cmds = (BULLNBC_Args_generic *) schedule->cmds;
    memcpy(cmds + idx, cmd, sizeof(BULLNBC_Args_generic));

    return OMPI_SUCCESS;
}

int BULLNBC_Sched_commit(BULLNBC_Schedule *schedule) {
//    if (!schedule->dynamic) {
        for (int i = 0; i < schedule->num_rounds; i++) {
            if (schedule->max_round_arity < schedule->arity[i])
                schedule->max_round_arity = schedule->arity[i];
            }
//    }

    schedule->requests = malloc(sizeof(MPI_Request) * schedule->max_round_arity);
    if (NULL == schedule->requests) {
        BULLNBC_release_schedule(schedule);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    schedule->rounds = malloc(sizeof(int) * schedule->num_rounds);;
    if (NULL == schedule->rounds) {
        BULLNBC_release_schedule(schedule);
        return OMPI_ERR_OUT_OF_RESOURCE;
    }

    if (OPAL_LIKELY(0 < schedule->num_rounds)){
        schedule->rounds[0] = 0;
        for (int i = 0; i < schedule->num_rounds - 1; i++) {
            schedule->rounds[i + 1] = schedule->rounds[i] + schedule->arity[i];
        }
    }

    return OMPI_SUCCESS;
}


int BULLNBC_Schedule_request(BULLNBC_Schedule *schedule, ompi_communicator_t *comm,
                         ompi_coll_bullnbc_module_t *module, bool persistent,
                         ompi_request_t **request, void *tmpbuf) {
    ompi_coll_bullnbc_request_t *handle;

    OMPI_COLL_BULLNBC_REQUEST_ALLOC(comm, persistent, handle);
    if (NULL == handle) {
        return OMPI_ERR_OUT_OF_RESOURCE;
    }
    handle->tmpbuf = NULL;
    handle->req_count = 0;
    handle->comm = comm;
    handle->schedule = NULL;
    handle->row_offset = 0;
    handle->nbc_complete = persistent ? true : false;
    /******************** Do the tag and shadow comm administration ...  ***************/

    handle->tag = ompi_coll_base_nbc_reserve_tags(comm, 1);
    module_need_progress(module);
    handle->comm=comm;

    /******************** end of tag and shadow comm administration ...  ***************/
    handle->comminfo = module;

    NBC_DEBUG(3, "got tag %i\n", handle->tag);

    handle->tmpbuf = tmpbuf;
    handle->schedule = (BULLNBC_Schedule *) schedule; /* dirty cast :) */
    handle->max_round = schedule->num_rounds;
    *request = (ompi_request_t *) handle;

    return OMPI_SUCCESS;
}

/* progresses a request
 *
 * to be called *only* from the progress thread !!! */
int BULLNBC_Progress(NBC_Handle *handle) {
	int ret = NBC_CONTINUE;
	bool flag;

	if (handle->nbc_complete) {
		return NBC_OK;
	}

	flag = true;

	if ((handle->req_count > 0) && (handle->req_array != NULL)) {
		NBC_DEBUG(50, "NBC_Progress: testing for %i requests\n", handle->req_count);

		/* don't call ompi_request_test_all as it causes a recursive call into opal_progress */
		while (handle->req_count) {
			ompi_request_t *subreq = handle->req_array[handle->req_count - 1];
			if (REQUEST_COMPLETE(subreq)) {
				if(OPAL_UNLIKELY( OMPI_SUCCESS != subreq->req_status.MPI_ERROR )) {
					NBC_Error ("MPI Error in NBC subrequest %p (req %p): %d", subreq, handle, subreq->req_status.MPI_ERROR);
					/* copy the error code from the underlying request and let the
					* round finish */
					handle->super.super.req_status.MPI_ERROR = subreq->req_status.MPI_ERROR;
				}
				handle->req_count--;
				ompi_request_free(&subreq);
			} else {
				flag = false;
				break;
			}
		}
	}

	/* a round is finished */
	if (flag) {
        int i, res;
        BULLNBC_Schedule *schedule;
        BULLNBC_Args_generic *cmds;

		handle->req_count = 0;
        schedule = (BULLNBC_Schedule *) handle->schedule;
        const int num = schedule->arity[handle->cur_round];

        cmds = (BULLNBC_Args_generic *) schedule->cmds;
        cmds = &(cmds[schedule->rounds[handle->cur_round]]);

        for (i = 0; i < num; i++) {
            switch (cmds[i].type) {
                case SEND:
                    if (cmds[i].args.send.retain) {
                        OBJ_RELEASE(cmds[i].args.send.datatype);
                    }
                    break;
                case RECV:
                    if (cmds[i].args.recv.retain) {
                        OBJ_RELEASE(cmds[i].args.recv.datatype);
                    }
                    break;
                case OP:
                    if (cmds[i].args.op.retain) {
                        OBJ_RELEASE(cmds[i].args.op.datatype);
                    }
                    break;
                case COPY:
                    if (cmds[i].args.copy.src_retain) {
                        OBJ_RELEASE(cmds[i].args.copy.srctype);
                    }
                    if (cmds[i].args.copy.tgt_retain) {
                        OBJ_RELEASE(cmds[i].args.copy.tgttype);
                    }
                    break;
                default:
                    /* TODO Pack/Unpack */
                    break;
                }
        }

		handle->cur_round++; /* prepare next round */

		/* previous round had an error */
		if (OPAL_UNLIKELY(OMPI_SUCCESS != handle->super.super.req_status.MPI_ERROR)) {
			res = handle->super.super.req_status.MPI_ERROR;
			NBC_Error("NBC_Progress: an error %d was found during schedule %p "
					  "at row-offset %li - aborting the schedule\n", res,
					  handle->schedule, handle->row_offset);
			handle->nbc_complete = true;
			if (!handle->super.super.req_persistent) {
				BULLNBC_Free(handle);
			}
			return res;
		}

		/* this was the last round - we're done */
		if (handle->cur_round >= handle->max_round) {
			/* bigger for an empty schedule */
			NBC_DEBUG(5, "NBC_Progress last round finished - we're done\n");
			handle->nbc_complete = true;
			if (!handle->super.super.req_persistent) {
				BULLNBC_Free(handle);
			}
			return NBC_OK;
		}

		NBC_DEBUG(5, "NBC_Progress round finished - goto next round\n");
		 /* kick it off */

		res = BULLNBC_Start_round(handle);
		if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
			NBC_Error ("Error in NBC_Start_round() (%i)", res);
			return res;
		}
	}

	return ret;
}

static inline int
BULLNBC_Start_round(NBC_Handle *handle) {
    int res;
    NBC_Args_send     sendargs;
    NBC_Args_recv     recvargs;
    NBC_Args_op         opargs;
    NBC_Args_copy     copyargs;
    NBC_Args_unpack unpackargs;

    BULLNBC_Args_generic *cmds;
    BULLNBC_Schedule *schedule;

    schedule = (BULLNBC_Schedule *)  handle->schedule; /* dirty cast :) */
    handle->req_array = schedule->requests;

    const int cround = handle->cur_round;
    const int num = schedule->arity[cround];

    cmds = (BULLNBC_Args_generic *) schedule->cmds;
    cmds = &(cmds[schedule->rounds[cround]]);

    for (int i = 0 ; i < num ; ++i) {
        const char *buf1;char *buf2;

        switch(cmds[i].type) {
            case SEND:
                sendargs = cmds[i].args.send;
                buf1 = sendargs.buf;
                buf1 += (sendargs.tmpbuf) ? (uintptr_t) handle->tmpbuf : 0;
                NBC_DEBUG(10, "[Tag%d] Start a send of count=%d, @%p, to %d\n",
                          handle->tag, sendargs.count,buf1,sendargs.dest);
                res = MCA_PML_CALL(isend(buf1, sendargs.count, sendargs.datatype,
                                   sendargs.dest, handle->tag,
                                   MCA_PML_BASE_SEND_STANDARD,
                                   sendargs.local ? handle->comm->c_local_comm : handle->comm,
                                   handle->req_array + handle->req_count));
                handle->req_count++;
                if (OMPI_SUCCESS != res) {
                    NBC_Error ("Error in MPI_Isend(%lu, %i, %p, %i, %i, %lu) (%i)",
                               (unsigned long)buf1, sendargs.count, sendargs.datatype,
                               sendargs.dest, handle->tag, (unsigned long) handle->comm, res);
                    return res;
                }
		        break;
            case RECV:
                recvargs = cmds[i].args.recv;
                buf2 = (void *) recvargs.buf;
                buf2 += (recvargs.tmpbuf) ? (uintptr_t) handle->tmpbuf : 0;
                NBC_DEBUG(10, "[Tag%d] Start a recv of count=%d, @%p, from  %d\n",
                          handle->tag, recvargs.count,buf2,recvargs.source);
                res = MCA_PML_CALL(irecv(buf2, recvargs.count, recvargs.datatype,
                                   recvargs.source, handle->tag,
                                   recvargs.local ? handle->comm->c_local_comm : handle->comm,
                                   handle->req_array + handle->req_count));
                handle->req_count++;
                if (OMPI_SUCCESS != res) {
                    NBC_Error("Error in MPI_Irecv(%lu, %i, %p, %i, %i, %lu) (%i)",
                              (unsigned long)buf2, recvargs.count, recvargs.datatype,
                              recvargs.source, handle->tag, (unsigned long) handle->comm, res);
                    return res;
                }
                break;
            case OP:
                opargs = cmds[i].args.op;
                buf1 = (void *) opargs.buf1;
                buf1 += (opargs.tmpbuf1) ? (uintptr_t) handle->tmpbuf : 0;
                buf2 = (void *) opargs.buf2;
                buf2 += (opargs.tmpbuf2) ? (uintptr_t) handle->tmpbuf : 0;
                NBC_DEBUG(5, "[Tag%d] Pop an OP : *src: %p, count: %i, type: %p, *tgt: %p)\n", handle->tag,
                          buf1, opargs.count, opargs.datatype, buf2);
                if (mca_coll_bullnbc_component.debug_read_user_data) {
                    for (int j = 0; j < opargs.count; ++j) {
                        NBC_DEBUG(15, "%x:", ((char *) buf1)[j]);
                    }
                    NBC_DEBUG(15, " (+) ");
                    for (int j = 0; j < opargs.count; ++j) {
                        NBC_DEBUG(15, "%x:", ((char *) buf2)[j]);
                    }
                    NBC_DEBUG(15, " = ");
                }
                ompi_op_reduce(opargs.op, (void*)buf1, buf2, opargs.count, opargs.datatype);
                if (mca_coll_bullnbc_component.debug_read_user_data) {
                    for (int j = 0; j < opargs.count; ++j) {
                        NBC_DEBUG(15, "%x:", ((char *) buf2)[j]);
                    }
                    NBC_DEBUG(15, "\n");
                }
                break;
            case COPY:
                copyargs = cmds[i].args.copy;
                buf1 = (void *) copyargs.src;
                buf1 += (copyargs.tmpsrc) ? (uintptr_t) handle->tmpbuf : 0;
                buf2 = (void *) copyargs.tgt;
                buf2 += (copyargs.tmptgt) ? (uintptr_t) handle->tmpbuf : 0;
                NBC_DEBUG(5, "[Tag%d] Pop an COPY *src: %p, srccount: %i, srctype: %p, *tgt: %p, tgtcount: %i, tgttype: %p)\n", handle->tag,
                            buf1, copyargs.srccount, copyargs.srctype,
                            buf2, copyargs.tgtcount, copyargs.tgttype);
                if (mca_coll_bullnbc_component.debug_read_user_data) {
                    for (int j = 0; j < copyargs.srccount; ++j) {
                        NBC_DEBUG(15, "%x:", ((char *) buf1)[j]);
                    }
                    NBC_DEBUG(15, "\n");
                }
                res = NBC_Copy (buf1, copyargs.srccount, copyargs.srctype, buf2, copyargs.tgtcount,
                                copyargs.tgttype, handle->comm);
                if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
                    return res;
                }
                break;
            case UNPACK:
                unpackargs = cmds[i].args.unpack;
                buf1 = (void *) unpackargs.inbuf;
                buf1 += (unpackargs.tmpinbuf) ? (uintptr_t) handle->tmpbuf : 0;
                buf2 = (void *) unpackargs.outbuf;
                buf2 += (unpackargs.tmpoutbuf) ? (uintptr_t) handle->tmpbuf : 0;
                res = NBC_Unpack (buf1, unpackargs.count, unpackargs.datatype, buf2, handle->comm);
                if (OMPI_SUCCESS != res) {
                    NBC_Error ("NBC_Unpack() failed (code: %i)", res);
                    return res;
                }
                break;

            default:
                NBC_Error ("NBC_Start_round: bad type %li at round: %d index: %d", (long)cmds[i].type, cround, i);
                return OMPI_ERROR;
        }
	}

    /* check if we can make progress - not in the first round, this allows us to leave the
     * initialization faster and to reach more overlap
     *
     * threaded case: calling progress in the first round can lead to a
     * deadlock if NBC_Free is called in this round :-( */
    if (0 && cround) {
        res = BULLNBC_Progress(handle);
        if ((NBC_OK != res) && (NBC_CONTINUE != res)) {
            return OMPI_ERROR;
        }
    }

    return OMPI_SUCCESS;
}

int BULLNBC_Start(NBC_Handle *handle) {
  int res;

  /* bozo case */
  if ((ompi_request_t *)handle == &ompi_request_empty) {
    return OMPI_SUCCESS;
  }

  /* kick off first round */
  handle->super.super.req_state = OMPI_REQUEST_ACTIVE;
  handle->super.super.req_status.MPI_ERROR = OMPI_SUCCESS;
  res = BULLNBC_Start_round(handle);
  if (OPAL_UNLIKELY(OMPI_SUCCESS != res)) {
    return res;
  }

  OPAL_THREAD_LOCK(&mca_coll_bullnbc_component.lock);
  opal_list_append(&mca_coll_bullnbc_component.active_requests, (opal_list_item_t *)handle);
  OPAL_THREAD_UNLOCK(&mca_coll_bullnbc_component.lock);

  return OMPI_SUCCESS;
}

static int BULLNBC_Sched_send_internal(const void* buf, char tmpbuf, int count,
                                       MPI_Datatype datatype, int dest, bool local,
                                       BULLNBC_Schedule *schedule, int round, int index) {
    int ret;
    BULLNBC_Args_generic gen_args;
  
    /* store the passed arguments */
    gen_args.type = SEND;
    gen_args.args.send.type = SEND;
    gen_args.args.send.buf = buf;
    gen_args.args.send.tmpbuf = tmpbuf;
    gen_args.args.send.count = count;
    gen_args.args.send.datatype = datatype;
    gen_args.args.send.dest = dest;
    gen_args.args.send.local = local;

    /* some algorithm create their own datatype */
    if (NULL != datatype && !ompi_datatype_is_predefined(datatype)) {
        gen_args.args.send.retain = true;
        OBJ_RETAIN(gen_args.args.send.datatype);
    } else {
        gen_args.args.send.retain = false;
    }

    /* append to the round-schedule */
    ret = BULLNBC_append_round_cmd(schedule, round, index, &gen_args);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    NBC_DEBUG(10, "added send - ends at round: %d idx: %d\n", round, index);

    return OMPI_SUCCESS;
}

int BULLNBC_Sched_send_insert (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                        int dest, BULLNBC_Schedule *schedule, int round, int index) {
    return BULLNBC_Sched_send_internal (buf, tmpbuf, count, datatype, dest,
                                        false, schedule, round, index);
}

int BULLNBC_Sched_send (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                        int dest, BULLNBC_Schedule *schedule, bool barrier) {
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_send_internal (buf, tmpbuf, count, datatype, dest,
                                        false, schedule, round, index);
}

#if 0
__opal_attribute_unused__
int BULLNBC_Sched_local_send_insert (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                                     int dest, BULLNBC_Schedule *schedule, int round, int index) {
    return BULLNBC_Sched_send_internal (buf, tmpbuf, count, datatype, dest,
                                        false, schedule, round, index);
}
#endif

int BULLNBC_Sched_local_send (const void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                              int dest, BULLNBC_Schedule *schedule, bool barrier) {
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_send_internal (buf, tmpbuf, count, datatype, dest,
                                        false, schedule, round, index);
}

static int BULLNBC_Sched_recv_internal (void* buf, char tmpbuf, int count,
                                    MPI_Datatype datatype, int source, bool local,
                                    BULLNBC_Schedule *schedule, int round, int index) {
    int ret;
    BULLNBC_Args_generic gen_args;
  
    /* store the passed arguments */
    gen_args.type = RECV;
    gen_args.args.recv.type = RECV;
    gen_args.args.recv.buf = buf;
    gen_args.args.recv.tmpbuf = tmpbuf;
    gen_args.args.recv.count = count;
    gen_args.args.recv.datatype = datatype;
    gen_args.args.recv.source = source;
    gen_args.args.recv.local = local;

    /* some algorithm create their own datatype */
    if (NULL != datatype && !ompi_datatype_is_predefined(datatype)) {
        gen_args.args.recv.retain = true;
        OBJ_RETAIN(gen_args.args.recv.datatype);
    } else {
        gen_args.args.recv.retain = false;
    }

    /* append to the round-schedule */
    ret = BULLNBC_append_round_cmd(schedule, round, index, &gen_args);
    if (OMPI_SUCCESS != ret) {
        return ret;
    }

    NBC_DEBUG(10, "added receive - ends at round: %d idx: %d\n", round, index);

    return OMPI_SUCCESS;
}

static inline int
BULLNBC_Sched_internal_barrier(BULLNBC_Schedule *schedule) {
    if (!schedule->dynamic) {
        return OMPI_SUCCESS;
    }

    if (schedule->max_round_arity < schedule->arity[schedule->num_rounds - 1]) {
        schedule->max_round_arity = schedule->arity[schedule->num_rounds - 1];
    }
    schedule->num_rounds++;
    return OMPI_SUCCESS;
}

static inline int
BULLNBC_Sched_internal_push(BULLNBC_Schedule *schedule, bool barrier, int *round, int *arity) {
    if (schedule->num_rounds >= schedule->max_rounds) {
        int i, *tmp;
        if (!schedule->dynamic) {
            return OMPI_ERROR;
        }
        schedule->max_rounds *= 2;
        tmp = (int *) realloc(schedule->arity, sizeof(int) * schedule->max_rounds);
        if (NULL == tmp) {
            return OMPI_ERR_OUT_OF_RESOURCE;
        }
        schedule->arity = tmp;
        for (i = schedule->num_rounds; i < schedule->max_rounds; i++) {
            schedule->arity[i] = 0;
        }
    }

    /* First round with cmd -- trig no ops */
    if (0 == schedule->num_rounds) {
        schedule->num_rounds = 1;
    }
    
    *arity = schedule->arity[schedule->num_rounds - 1]++;
    *round = schedule->num_rounds -1;

    if (barrier) {
        (void) BULLNBC_Sched_internal_barrier(schedule);
    }
    
    return OMPI_SUCCESS;
}

int BULLNBC_Sched_recv_insert (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                               int source, BULLNBC_Schedule *schedule, int round, int index) 
{
  return BULLNBC_Sched_recv_internal(buf, tmpbuf, count, datatype, source,
                                 false, schedule, round, index);
}

int BULLNBC_Sched_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                        int source, BULLNBC_Schedule *schedule, bool barrier)
{
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_recv_internal(buf, tmpbuf, count, datatype, source,
                                       false, schedule, round, index);
}

#if 0
__opal_attribute_unused__
int BULLNBC_Sched_local_recv_insert (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                                     int source, BULLNBC_Schedule *schedule, int round, int index)
{
    return BULLNBC_Sched_recv_internal(buf, tmpbuf, count, datatype, source,
                                       true, schedule, round, index);
}
#endif


int BULLNBC_Sched_local_recv (void* buf, char tmpbuf, int count, MPI_Datatype datatype,
                          int source, BULLNBC_Schedule *schedule, bool barrier) {
    int round=0, index=0;
    (void) BULLNBC_Sched_internal_push(schedule, barrier, &round, &index);
    return BULLNBC_Sched_recv_internal(buf, tmpbuf, count, datatype, source,
                                       true, schedule, round, index);
}

int BULLNBC_Sched_barrier(BULLNBC_Schedule *schedule) {
    return BULLNBC_Sched_internal_barrier(schedule);
}

BULLNBC_Schedule *BULLNBC_create_schedule(int nrounds, int *round_arity, bool dynamic) {
    BULLNBC_Schedule *new_schedule;

    new_schedule = (BULLNBC_Schedule*) opal_free_list_wait (&mca_coll_bullnbc_component.schedules);
    new_schedule->dynamic = dynamic;
    new_schedule->max_rounds = BULLNBC_DEFAULT_MIN_ROUND;
    if (new_schedule->max_rounds < nrounds)
        new_schedule->max_rounds = nrounds;

    new_schedule->num_rounds = (dynamic) ? 0 : nrounds;
    new_schedule->max_cmds = BULLNBC_DEFAULT_MIN_CMD;

    new_schedule->num_cmds = 0;
    new_schedule->max_round_arity = -1;

    new_schedule->requests = NULL;
    new_schedule->rounds = NULL;

    if (!dynamic) {
        int i, ncmds;

        for (ncmds = 0, i = 0; i < nrounds; i++) {
           ncmds += round_arity[i];
        }

        if (new_schedule->max_cmds < ncmds) {
           new_schedule->max_cmds = ncmds;
        } 
    }

    new_schedule->cmds = (void *) malloc(sizeof(BULLNBC_Args_generic) * new_schedule->max_cmds);
    if (NULL == new_schedule->cmds) {
        BULLNBC_release_schedule(new_schedule);
        return NULL;
    }

    new_schedule->arity = malloc(sizeof(int) * new_schedule->max_rounds);
    if (NULL == new_schedule->arity) {
        BULLNBC_release_schedule(new_schedule);
        return NULL;
    }

    if (!dynamic) {
        memcpy(new_schedule->arity, round_arity, sizeof(int) * nrounds);
    } else {
        memset(new_schedule->arity, 0, sizeof(int) * new_schedule->max_rounds);
    }

    return new_schedule;
}
