#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

# TODO: unused parameters to remove once missing functions are implemented
AM_CFLAGS = $(opal_mpc_lowcomm_CPPFLAGS) -Werror
AM_LDFLAGS = $(opal_mpc_lowcomm_LDFLAGS) $(opal_mpc_lowcomm_LIBS)

if MCA_BUILD_ompi_pml_mpc_DSO
component_noinst =
component_install = mca_pml_mpc.la
else
component_noinst = libmca_pml_mpc.la
component_install =
endif

local_sources = \
	pml_mpc.c \
	pml_mpc.h \
	pml_mpc_pack.h \
	pml_mpc_pack.c \
	pml_mpc_component.c \
	pml_mpc_irecv.c \
	pml_mpc_isend.c \
        pml_mpc_request.c \
        pml_mpc_request.h \
        pml_mpc_communicator.h \
        pml_mpc_communicator.c

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_pml_mpc_la_SOURCES = $(local_sources)
mca_pml_mpc_la_LDFLAGS = -module -avoid-version $(AM_LDFLAGS)
mca_pml_mpc_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_pml_mpc_la_SOURCES = $(local_sources)
libmca_pml_mpc_la_LDFLAGS = -module -avoid-version $(AM_LDFLAGS)

