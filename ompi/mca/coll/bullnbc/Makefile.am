#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2009 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2010      Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2013      Los Alamos National Security, LLC. All rights
#                         reserved.
# Copyright (c) 2017      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2017      IBM Corporation.  All rights reserved.
# Copyright (c) 2021-2022 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

dist_ompidata_DATA = help-mpi-coll-bullnbc.txt

sources = \
        coll_bullnbc.h \
	coll_bullnbc_component.c \
	coll_bullnbc.c \
	coll_bullnbc_internal.h \
	coll_bullnbc_iallgather.c \
	coll_bullnbc_iallgatherv.c \
	coll_bullnbc_iallreduce.c \
	coll_bullnbc_ialltoall.c \
	coll_bullnbc_ialltoallv.c \
	coll_bullnbc_ialltoallw.c \
	coll_bullnbc_ibarrier.c \
	coll_bullnbc_ibcast.c \
	coll_bullnbc_iexscan.c \
	coll_bullnbc_igather.c \
	coll_bullnbc_igatherv.c \
	coll_bullnbc_igatherw.c \
	coll_bullnbc_ineighbor_allgather.c \
	coll_bullnbc_ineighbor_allgatherv.c \
	coll_bullnbc_ineighbor_alltoall.c \
	coll_bullnbc_ineighbor_alltoallv.c \
	coll_bullnbc_ineighbor_alltoallw.c \
	coll_bullnbc_ireduce.c \
	coll_bullnbc_ireduce_scatter.c \
	coll_bullnbc_ireduce_scatter_block.c \
	coll_bullnbc_iscan.c \
	coll_bullnbc_iscatter.c \
	coll_bullnbc_iscatterv.c \
	coll_bullnbc_neighbor_helpers.c \
	coll_bullnbc_partitioned.c \
	coll_bullnbc_partitioned.h \
	coll_bullnbc_partitioned_noop.c \
	coll_bullnbc_partitioned_dag.c \
	coll_bullnbc_partitioned_dag_sched.c \
	coll_bullnbc_partitioned_dag.h \
	base/coll_bullnbc_base_ibarrier.c \
	base/coll_bullnbc_base.h

# Make the output library in this directory, and name it either
# mca_<type>_<name>.la (for DSO builds) or libmca_<type>_<name>.la
# (for static builds).

if MCA_BUILD_ompi_coll_bullnbc_DSO
component_noinst =
component_install = mca_coll_bullnbc.la
else
component_noinst = libmca_coll_bullnbc.la
component_install =
endif

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_coll_bullnbc_la_SOURCES = $(sources)
mca_coll_bullnbc_la_LDFLAGS = -module -avoid-version
mca_coll_bullnbc_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

noinst_LTLIBRARIES = $(component_noinst)
libmca_coll_bullnbc_la_SOURCES =$(sources)
libmca_coll_bullnbc_la_LDFLAGS = -module -avoid-version
