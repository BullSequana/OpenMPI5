! -*- f90 -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$

#include "ompi/mpi/fortran/configure-fortran-output.h"

subroutine PMPIX_Win_create_notify_f08(base,size,disp_unit,info,comm,win,ierror)
   use :: mpi_f08_types, only : MPI_Info, MPI_Comm, MPI_Win, MPI_ADDRESS_KIND
   implicit none
   interface
         subroutine ompix_win_create_notify_f(base,size,disp_unit,info,comm,win,ierror) &
         BIND(C, name="ompix_win_create_notify_f")
         use :: mpi_f08_types, only : MPI_ADDRESS_KIND
         implicit none
         OMPI_FORTRAN_IGNORE_TKR_TYPE, INTENT(IN) :: base
         INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: size
         INTEGER, INTENT(IN) :: disp_unit
         INTEGER, INTENT(IN) :: info
         INTEGER, INTENT(IN) :: comm
         INTEGER, INTENT(OUT) :: win
         INTEGER, INTENT(OUT) :: ierror
         end subroutine ompix_win_create_notify_f
   end interface
   OMPI_FORTRAN_IGNORE_TKR_TYPE :: base
   INTEGER(MPI_ADDRESS_KIND), INTENT(IN) :: size
   INTEGER, INTENT(IN) :: disp_unit
   TYPE(MPI_Info), INTENT(IN) :: info
   TYPE(MPI_Comm), INTENT(IN) :: comm
   TYPE(MPI_Win), INTENT(OUT) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompix_win_create_notify_f(base,size,disp_unit,info%MPI_VAL,&
                                 comm%MPI_VAL,win%MPI_VAL,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPIX_Win_create_notify_f08
