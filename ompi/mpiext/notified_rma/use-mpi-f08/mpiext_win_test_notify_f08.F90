! -*- f90 -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$

subroutine MPIX_Win_test_notify_f08(win,flag,notification_id,ierror)
   use :: mpi_f08_types, only : MPI_Win
   implicit none
   interface
      subroutine ompix_win_test_notify_f(win,flag,notification_id,ierror) &
         BIND(C, name="ompix_win_test_notify_f")
        implicit none
        LOGICAL, INTENT(OUT) :: flag
        INTEGER, INTENT(IN) :: notification_id
        INTEGER, INTENT(IN) :: win
        INTEGER, INTENT(OUT) :: ierror
      end subroutine ompix_win_test_notify_f
   end interface
   LOGICAL, INTENT(OUT) :: flag
   INTEGER, INTENT(IN) :: notification_id
   TYPE(MPI_Win), INTENT(IN) :: win
   INTEGER, OPTIONAL, INTENT(OUT) :: ierror
   integer :: c_ierror

   call ompix_win_test_notify_f(win%MPI_VAL,flag,notification_id,c_ierror)
   if (present(ierror)) ierror = c_ierror
end subroutine MPIX_Win_test_notify_f08
