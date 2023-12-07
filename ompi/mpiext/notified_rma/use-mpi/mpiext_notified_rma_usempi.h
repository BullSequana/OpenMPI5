! -*- fortran -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!

! This whole file will be included in the mpi_ext module interface
! section.  Note that the extension's mpif.h file will be included
! first, so there's no need to re-define anything that's in there (e.g.,
! OMPI_EXAMPLE_GLOBAL).

! Declare any interfaces, subroutines, and global variables/constants
! here.  Note that the mpiext_notified_rma_mpif.h will automatically be
! included before this, so anything declared there does not need to be
! replicated here.
!
LOGICAL OMPI_HAVE_MPI_EXT_NOTIFIED_RMA
    parameter (OMPI_HAVE_MPI_EXT_NOTIFIED_RMA = .TRUE.)

interface MPIX_Get_notify
    subroutine MPIX_Get_notify(origin_addr, origin_count, origin_datatype, target_rank, target_disp, &
            target_count, target_datatype, win, notification_id, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !$PRAGMA IGNORE_TKR origin_addr
      !DIR$ IGNORE_TKR origin_addr
      !IBM* IGNORE_TKR origin_addr
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: origin_addr
      integer, intent(in) :: origin_count
      integer, intent(in) :: origin_datatype
      integer, intent(in) :: target_rank
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: target_disp
      integer, intent(in) :: target_count
      integer, intent(in) :: target_datatype
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine MPIX_Get_notify
end interface MPIX_Get_notify

interface MPIX_Put_notify
    subroutine MPIX_Put_notify(origin_addr, origin_count, origin_datatype, target_rank, target_disp, &
            target_count, target_datatype, win, notification_id, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !$PRAGMA IGNORE_TKR origin_addr
      !DIR$ IGNORE_TKR origin_addr
      !IBM* IGNORE_TKR origin_addr
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: origin_addr
      integer, intent(in) :: origin_count
      integer, intent(in) :: origin_datatype
      integer, intent(in) :: target_rank
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: target_disp
      integer, intent(in) :: target_count
      integer, intent(in) :: target_datatype
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine MPIX_Put_notify
end interface MPIX_Put_notify

interface MPIX_Win_allocate_notify
    subroutine MPIX_Win_allocate_notify(size, disp_unit, info, comm, &
          baseptr, win, ierror)
      include 'mpif-config.h'
      integer(KIND=MPI_ADDRESS_KIND), intent(in) :: size
      integer, intent(in) :: disp_unit
      integer, intent(in) :: info
      integer, intent(in) :: comm
      integer(KIND=MPI_ADDRESS_KIND), intent(out) :: baseptr
      integer, intent(out) :: win
      integer, intent(out) :: ierror
    end subroutine MPIX_Win_allocate_notify
end interface MPIX_Win_allocate_notify

interface MPIX_Win_create_notify
    subroutine MPIX_Win_create_notify(base, size, disp_unit, info, comm, &
            win, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: base
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: base
      !$PRAGMA IGNORE_TKR base
      !DIR$ IGNORE_TKR base
      !IBM* IGNORE_TKR base
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: base
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: size
      integer, intent(in) :: disp_unit
      integer, intent(in) :: info
      integer, intent(in) :: comm
      integer, intent(out) :: win
      integer, intent(out) :: ierror
    end subroutine MPIX_Win_create_notify
end interface MPIX_Win_create_notify

interface MPIX_Win_test_notify
    subroutine MPIX_Win_test_notify(win, flag, notification_id, ierror)
      integer, intent(in) :: win
      logical, intent(out) :: flag
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine MPIX_Win_test_notify
end interface MPIX_Win_test_notify

interface MPIX_Win_wait_notify
    subroutine MPIX_Win_wait_notify(win, notification_id, ierror)
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine MPIX_Win_wait_notify
end interface MPIX_Win_wait_notify

interface PMPIX_Get_notify
    subroutine PMPIX_Get_notify(origin_addr, origin_count, origin_datatype, target_rank, target_disp, &
            target_count, target_datatype, win, notification_id, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !$PRAGMA IGNORE_TKR origin_addr
      !DIR$ IGNORE_TKR origin_addr
      !IBM* IGNORE_TKR origin_addr
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: origin_addr
      integer, intent(in) :: origin_count
      integer, intent(in) :: origin_datatype
      integer, intent(in) :: target_rank
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: target_disp
      integer, intent(in) :: target_count
      integer, intent(in) :: target_datatype
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine PMPIX_Get_notify
end interface PMPIX_Get_notify

interface PMPIX_Put_notify
    subroutine PMPIX_Put_notify(origin_addr, origin_count, origin_datatype, target_rank, target_disp, &
            target_count, target_datatype, win, notification_id, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: origin_addr
      !$PRAGMA IGNORE_TKR origin_addr
      !DIR$ IGNORE_TKR origin_addr
      !IBM* IGNORE_TKR origin_addr
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: origin_addr
      integer, intent(in) :: origin_count
      integer, intent(in) :: origin_datatype
      integer, intent(in) :: target_rank
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: target_disp
      integer, intent(in) :: target_count
      integer, intent(in) :: target_datatype
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine PMPIX_Put_notify
end interface PMPIX_Put_notify

interface PMPIX_Win_allocate_notify
    subroutine PMPIX_Win_allocate_notify(size, disp_unit, info, comm, &
          baseptr, win, ierror)
      include 'mpif-config.h'
      integer(KIND=MPI_ADDRESS_KIND), intent(in) :: size
      integer, intent(in) :: disp_unit
      integer, intent(in) :: info
      integer, intent(in) :: comm
      integer(KIND=MPI_ADDRESS_KIND), intent(out) :: baseptr
      integer, intent(out) :: win
      integer, intent(out) :: ierror
    end subroutine PMPIX_Win_allocate_notify
end interface PMPIX_Win_allocate_notify

interface PMPIX_Win_create_notify
    subroutine PMPIX_Win_create_notify(base, size, disp_unit, info, comm, &
            win, ierror)
      include 'mpif-config.h'
      !DEC$ ATTRIBUTES NO_ARG_CHECK :: base
      !GCC$ ATTRIBUTES NO_ARG_CHECK :: base
      !$PRAGMA IGNORE_TKR base
      !DIR$ IGNORE_TKR base
      !IBM* IGNORE_TKR base
      OMPI_FORTRAN_IGNORE_TKR_TYPE, intent(in) :: base
      integer(kind=MPI_ADDRESS_KIND), intent(in) :: size
      integer, intent(in) :: disp_unit
      integer, intent(in) :: info
      integer, intent(in) :: comm
      integer, intent(out) :: win
      integer, intent(out) :: ierror
    end subroutine PMPIX_Win_create_notify
end interface PMPIX_Win_create_notify

interface PMPIX_Win_test_notify
    subroutine PMPIX_Win_test_notify(win, flag, notification_id, ierror)
      integer, intent(in) :: win
      logical, intent(out) :: flag
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine PMPIX_Win_test_notify
end interface PMPIX_Win_test_notify

interface PMPIX_Win_wait_notify
    subroutine PMPIX_Win_wait_notify(win, notification_id, ierror)
      integer, intent(in) :: win
      integer, intent(in) :: notification_id
      integer, intent(out) :: ierror
    end subroutine PMPIX_Win_wait_notify
end interface PMPIX_Win_wait_notify
