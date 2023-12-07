#
# Copyright (c) 2004-2009 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2011      Oak Ridge National Labs.  All rights reserved.
# Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST).  All rights reserved.
# Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This file builds the mpif.h-based bindings for MPI extensions.  It
# is optional in MPI extensions.

# We must set these #defines so that the inner OMPI MPI prototype
# header files do the Right Thing.
AM_CPPFLAGS = -DOMPI_PROFILE_LAYER=0 -DOMPI_COMPILING_FORTRAN_WRAPPERS=1

# Note that the mpif.h-based bindings are optional -- they can only be
# built if OMPI is also building the Fortran-based bindings.  So we
# initially set some Makefile macros to empty, and then conditionally
# add to them later.
noinst_LTLIBRARIES =

# Directory where the header file to be included in mpif-ext.h must be
# installed.
ompidir = $(ompiincludedir)/mpiext

# Just like noinst_LTLIBRARIES, set this macro to empty and
# conditionally add to it later.
ompi_HEADERS =

# Use the Automake conditional to know if we're building the mpif.h
# bindings.
if OMPI_BUILD_FORTRAN_MPIFH_BINDINGS

# If we are, build the convenience libtool library that will be
# slurped up into libmpi_mpifh.la.
noinst_LTLIBRARIES += libmpiext_notified_rma_mpifh.la

# This is the header file that is installed.
ompi_HEADERS += mpiext_notified_rma_mpifh.h
noinst_HEADERS = mpiext_notified_rma_prototypes.h

# Sources for the convenience libtool library.
libmpiext_notified_rma_mpifh_la_SOURCES = \
        $(ompi_HEADERS) \
		mpiext_get_notify_f.c \
		mpiext_put_notify_f.c \
		mpiext_win_allocate_notify_f.c \
		mpiext_win_create_notify_f.c \
		mpiext_win_test_notify_f.c \
		mpiext_win_wait_notify_f.c
libmpiext_notified_rma_mpifh_la_LDFLAGS = -module -avoid-version
endif
