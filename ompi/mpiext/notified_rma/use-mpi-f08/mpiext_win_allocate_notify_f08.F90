! -*- f90 -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$

subroutine MPIX_Win_allocate_notify_f08(size, disp_unit, info, comm, &
      baseptr, win, ierror)
   USE, INTRINSIC ::  ISO_C_BINDING, ONLY : C_PTR
   use :: mpi_f08_types, only : MPI_Info, MPI_Comm, MPI_Win, MPI_ADDRESS_KIND
   implicit none
   interface
        subroutine ompix_win_allocate_notify_f(size, disp_unit, info, comm, &
                baseptr, win, ierror) BIND(C, name="ompix_win_allocate_notify_f")
            USE, INTRINSIC ::  ISO_C_BINDING, ONLY : C_PTR
            use :: mpi_f08_types, only : MPI_ADDRESS_KIND
            INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) ::  size
            INTEGER, INTENT(IN) ::  disp_unit
            INTEGER, INTENT(IN) ::  info
            INTEGER, INTENT(IN) ::  comm
            TYPE(C_PTR), INTENT(OUT) ::  baseptr
            INTEGER, INTENT(OUT) ::  win
            INTEGER, INTENT(OUT) ::  ierror
        end subroutine ompix_win_allocate_notify_f
   end interface
   INTEGER(KIND=MPI_ADDRESS_KIND), INTENT(IN) ::  size
   INTEGER, INTENT(IN) ::  disp_unit
   TYPE(MPI_Info), INTENT(IN) ::  info
   TYPE(MPI_Comm), INTENT(IN) ::  comm
   TYPE(C_PTR), INTENT(OUT) ::  baseptr
   TYPE(MPI_Win), INTENT(OUT) ::  win
   INTEGER, OPTIONAL, INTENT(OUT) ::  ierror
   integer :: c_ierror

  call ompix_win_allocate_notify_f(size, disp_unit, info%MPI_VAL, comm%MPI_VAL, baseptr, win%MPI_VAL, c_ierror)
   if (present(ierror)) ierror = c_ierror

end subroutine MPIX_Win_allocate_notify_f08
