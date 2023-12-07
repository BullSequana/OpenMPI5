! -*- f90 -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$

subroutine PMPIX_Win_wait_notify_f08(win,notification_id,ierror)
   use :: mpi_f08_types, only : MPI_Win
   implicit none
   interface
      subroutine ompix_win_wait_notify_f(win,notification_id,ierror) &
         BIND(C, name="ompix_win_wait_notify_f")
        implicit none
        INTEGER, INTENT(IN) :: notification_id
        INTEGER, INTENT(IN) :: win
        INTEGER, INTENT(OUT) :: ierror
      end subroutine ompix_win_wait_notify_f
   end interface
   INTEGER, INTENT(IN) :: notification_id
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompix_win_wait_notify_f(win%MPI_VAL,notification_id,c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine PMPIX_Win_wait_notify_f08
