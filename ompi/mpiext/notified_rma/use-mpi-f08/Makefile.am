#
# Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This file builds the use_mpi_f08-based bindings for MPI extensions.  It
# is optional in MPI extensions.

# Note that Automake's Fortran-buidling rules uses CPPFLAGS and
# AM_CPPFLAGS.  This can cause weirdness (e.g.,
# https://github.com/open-mpi/ompi/issues/7253 and
# https://github.com/open-mpi/ompi/issues/9716).  Let's just zero
# those out and rely on AM_FCFLAGS.
CPPFLAGS =
AM_CPPFLAGS =

# We must set these #defines and include paths so that the inner OMPI
# MPI prototype header files do the Right Thing.
AM_FCFLAGS = $(OMPI_FC_MODULE_FLAG)$(top_builddir)/ompi/mpi/fortran/use-mpi \
             $(OMPI_FC_MODULE_FLAG)$(top_builddir)/ompi/mpi/fortran/use-mpi-f08/mod \
             -I$(top_builddir) -I$(top_srcdir) $(FCFLAGS_f90)

# Note that the mpi_f08-based bindings are optional -- they can only
# be built if OMPI is also building the Fortran-based bindings.  So we
# initially set some Makefile macros to empty, and then conditionally
# add to them later.
noinst_LTLIBRARIES =

# Use the Automake conditional to know if we're building the mpif.h
# bindings.
if OMPI_BUILD_FORTRAN_USEMPIF08_BINDINGS

# If we are, build the convenience libtool library that will be
# slurped up into libmpi_usempif08.la.
noinst_LTLIBRARIES += libmpiext_notified_rma_usempif08.la

# Note that no header files are installed; instead,
# mpiext_example_usempif08.h is automatically slurped up into the
# mpi_f08_ext module.  It must be listed so that it is included in
# dist tarballs.
noinst_HEADERS = mpiext_notified_rma_usempif08.h

# Sources for the convenience libtool library.
libmpiext_notified_rma_usempif08_la_SOURCES = mpiext_get_notify_f08.F90 \
						  					  mpiext_put_notify_f08.F90 \
											  mpiext_win_allocate_notify_f08.F90 \
											  mpiext_win_create_notify_f08.F90 \
											  mpiext_win_test_notify_f08.F90 \
											  mpiext_win_wait_notify_f08.F90 \
                                              profile/pmpiext_get_notify_f08.F90 \
						  					  profile/pmpiext_put_notify_f08.F90 \
											  profile/pmpiext_win_allocate_notify_f08.F90 \
											  profile/pmpiext_win_create_notify_f08.F90 \
											  profile/pmpiext_win_test_notify_f08.F90 \
											  profile/pmpiext_win_wait_notify_f08.F90 
libmpiext_notified_rma_usempif08_la_LDFLAGS = -module -avoid-version
endif
