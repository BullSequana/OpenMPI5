#
# Copyright (c) 2004-2009 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2010-2014 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2015      NVIDIA, Inc. All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST).  All rights reserved.
# Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This file builds the C bindings for MPI extensions.  It must be
# present in all MPI extensions.

# We must set these #defines so that the inner OMPI MPI prototype
# header files do the Right Thing.
AM_CPPFLAGS = -DOMPI_PROFILE_LAYER=0 -DOMPI_COMPILING_FORTRAN_WRAPPERS=1

dist_ompidata_DATA = help-mpi-notifications.txt
include $(top_srcdir)/Makefile.ompi-rules

# Convenience libtool library that will be slurped up into libmpi.la.
noinst_LTLIBRARIES = libmpiext_notified_rma_c.la

# This is where the top-level header file (that is included in
# <mpi-ext.h>) must be installed.
ompidir = $(ompiincludedir)/mpiext

# This is the header file that is installed.
nodist_ompi_HEADERS = mpiext_notified_rma_c.h

# Internal header that must not be installed
noinst_HEADERS = mpiext_notified_rma_c.h
noinst_HEADERS += mpiext_notifs_rma.h

# Sources for the convenience libtool library. Other than the one
# header file, all source files in the extension have no file naming
# conventions.
libmpiext_notified_rma_c_la_SOURCES = \
        $(ompi_HEADERS) \
        mpiext_get_notify.c \
        mpiext_put_notify.c \
        mpiext_win_allocate_notify.c \
        mpiext_win_create_notify.c \
        mpiext_win_test_notify.c \
        mpiext_win_wait_notify.c \
		mpiext_notified_rma.c
libmpiext_notified_rma_c_la_LDFLAGS = -module -avoid-version

# Man page installation
nodist_man_MANS =
#nodist_man_MANS = MPIX_Get_notify.3 \
#				  MPIX_Put_notify.3 \
#				  MPIX_Win_allocate_notify.3 \
#				  MPIX_Win_create_notify.3 \
#				  MPIX_Win_test_notify.3 \
#				  MPIX_Win_wait_notify.3

# Man page sources
EXTRA_DIST = $(nodist_man_MANS:.3=.3in)

distclean-local:
	rm -f $(nodist_man_MANS)
