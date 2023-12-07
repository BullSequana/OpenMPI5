/*
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 *
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 *
 */

OMPI_DECLSPEC int MPIX_Get_notify(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, int notification_id);
OMPI_DECLSPEC int MPIX_Put_notify(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, int notification_id);
OMPI_DECLSPEC int MPIX_Win_allocate_notify(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win);
OMPI_DECLSPEC int MPIX_Win_create_notify(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
OMPI_DECLSPEC int MPIX_Win_test_notify(MPI_Win win, int notification_id, int *flag);
OMPI_DECLSPEC int MPIX_Win_wait_notify(MPI_Win win, int notification_id);

/* Profiling MPI API */

OMPI_DECLSPEC int PMPIX_Get_notify(void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank, MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, int notification_id);
OMPI_DECLSPEC int PMPIX_Put_notify(const void *origin_addr, int origin_count, MPI_Datatype origin_datatype, int target_rank,MPI_Aint target_disp, int target_count, MPI_Datatype target_datatype, MPI_Win win, int notification_id);
OMPI_DECLSPEC int PMPIX_Win_allocate_notify(MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, void *baseptr, MPI_Win *win);
OMPI_DECLSPEC int PMPIX_Win_create_notify(void *base, MPI_Aint size, int disp_unit, MPI_Info info, MPI_Comm comm, MPI_Win *win);
OMPI_DECLSPEC int PMPIX_Win_test_notify(MPI_Win win, int notification_id, int *flag);
OMPI_DECLSPEC int PMPIX_Win_wait_notify(MPI_Win win, int notification_id);
