
PML MPC uses MPC Lowcomm interface: low level messaging interface of MPC.


Tasks:
- [x] send / recv
- [x] isend / irecv using requests & pending list: see pml_mpc_request.c
- [x] progress using new mpc_lowcomm_test()
- [x] verbosity
- [x] OSU p2p
- [x] set CFLAGS & LDFLAGS (manually)
- [x] completion callback in MPC
- [x] use -Werror. must ignore unused-parameters for unimplemented functions
- [x] build: MPC detection in m4
- [x] MPI Status set on completion
- [x] free-list for pending list
- [x] Reformating with OMPI rules
- [ ] check request/OBJ usage (init should be called auto ?)
- [x] alltoall, needing persistant requests (isend_init())
- [x] MPI_COMM_SELF, MPI_COMM_NULL -> MPC communicators
- [x] new fake communicators (e.g. for IMB) but with WORLD as parent
- [x] IMB (using fake communicators + ANY_SOURCE)
  [x] communicator using MPC communicator
- [ ] right parent for mpc communicators
- [x] non-contiguous datatypes using pack unpack
- [x] remove -Wno-unused-parameters (adding TODO)
- [ ] mprobe/mrecv
- [ ] send modes
  [ ] include tests inside module
  [ ] -Werror complaints on mpc files

Bugs:
- [x] completion hang: use ompi_request_complete instead of flag
- [x] request double-free: reset to MPI_REQUEST_NULL
- [x] detection of includes in include/mpcframework/
- [x] completion hang when using completion callback in MPC -> request re-initialized
- [x] IMB hang on communicator creation with procs>2 -> ANY_SOURCE
- [ ] IMB reduce_local check ko : not in PML_MPC (same with UCX)
