! -*- fortran -*-
!
! Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
! $COPYRIGHT$
!
! Additional copyrights may follow
!
! $HEADER$
!

! This whole file will be included in the mpi_f08_ext module interface
! section.  Note that the extension's mpif.h file will be included
! first, so there's no need to re-define anything that's in there.

! Declare any interfaces, subroutines, and global variables/constants
! here.  Note that the mpiext_notified_rma_mpif.h will automatically be
! included before this, so anything declared there does not need to be
! replicated here.

LOGICAL OMPI_HAVE_MPI_EXT_NOTIFIED_RMA
    parameter (OMPI_HAVE_MPI_EXT_NOTIFIED_RMA = .TRUE.)

