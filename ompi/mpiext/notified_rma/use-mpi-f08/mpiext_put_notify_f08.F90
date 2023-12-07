! -*- f90 -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine MPIX_Put_notify_f08(origin_addr,origin_count,origin_datatype,target_rank,&
                               target_disp,target_count,target_datatype,win,notification_id,ierror)
   use :: mpi_f08_types, only : MPI_Datatype, MPI_Win, MPI_ADDRESS_KIND
   implicit none
   interface
        subroutine ompix_put_notify_f(origin_addr,origin_count,origin_datatype,target_rank, &
                              target_disp,target_count,target_datatype,win,notification_id,ierror) &
             BIND(C, name="ompix_put_notify_f")
            use :: mpi_f08_types, only : MPI_ADDRESS_KIND
            implicit none
            OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN) :: origin_addr
            INTEGER, INTENT(IN) :: origin_count, target_rank, target_count, notification_id
            INTEGER, INTENT(IN) :: origin_datatype
            INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
            INTEGER, INTENT(IN) :: target_datatype
            INTEGER, INTENT(IN) :: win
            INTEGER, INTENT(OUT) :: ierror
        end subroutine ompix_put_notify_f
   end interface
   OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN), ASYNCHRONOUS :: origin_addr
   INTEGER, INTENT(IN) :: origin_count, target_rank, target_count, notification_id
   TYPE(MPI_Datatype), INTENT(IN) :: origin_datatype
   INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: target_disp
   TYPE(MPI_Datatype), INTENT(IN) :: target_datatype
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompix_put_notify_f(origin_addr,origin_count,origin_datatype%MPI_VAL,target_rank,&
                           target_disp,target_count,target_datatype%MPI_VAL,win%MPI_VAL,notification_id,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPIX_Put_notify_f08
