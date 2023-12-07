#
# Copyright (c) 2017      FUJITSU LIMITED.  All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

#SUBDIRS = profile

# TODO REMOVE ME ??
#
# OMPI_BUILD_MPI_PROFILING is enabled when we want our generated MPI_* symbols
# to be replaced by PMPI_*.
# In this directory, we need it to be 0

include $(top_srcdir)/Makefile.ompi-rules

noinst_LTLIBRARIES = libmpiext_pcoll_part_c.la

# This is where the top-level header file (that is included in
# <mpi-ext.h>) must be installed.
ompidir = $(ompiincludedir)/mpiext

# This is the header file that is installed.
ompi_HEADERS = mpiext_pcoll_part_c.h
# Internal header that must not be installed
# noinst_HEADERS =

# Sources for the convenience libtool library.  Other than the one
# header file, all source files in the extension have no file naming
# conventions.
libmpiext_pcoll_part_c_la_SOURCES = \
        $(ompi_HEADERS)             \
        palltoall_init.c            \
        palltoallr_init.c           \
        palltoallv_init.c           \
        palltoallvr_init.c          \
        pbcast_init.c               \
        pbcastr_init.c              \
        pgather_init.c              \
        pgatherr_init.c             \
        preduce_init.c              \
        preducer_init.c

# Man page installation
nodist_man_MANS =
#nodist_man_MANS = \
#        MPIX_Palltoall_init.3       \
#        MPIX_Palltoallr_init.3      \
#        MPIX_Palltoallv_init.3      \
#        MPIX_Palltoallvr_init.3     \
#        MPIX_Pbcast_init.3          \
#        MPIX_Pbcastr_init.3         \
#        MPIX_Pgather_init.3         \
#        MPIX_Pgatherr_init.3        \
#        MPIX_Preduce_init.3         \
#        MPIX_Preducer_init.3

# Man page sources
EXTRA_DIST = $(nodist_man_MANS:.3=.3in)

distclean-local:
	rm -f $(nodist_man_MANS)
