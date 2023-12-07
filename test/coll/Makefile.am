#
# Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

AM_CPPFLAGS = -I$(top_srcdir)/test/support -DRULES_FILE_DIR=$(top_srcdir)/contrib/platform/bull/profile

check_PROGRAMS = \
	ompi_coll_configfile

TESTS = \
	$(check_PROGRAMS)

ompi_coll_configfile_SOURCES = ompi_coll_configfile.c
ompi_coll_configfile_LDADD = \
        $(top_builddir)/opal/lib@OPAL_LIB_PREFIX@open-pal.la \
        $(top_builddir)/test/support/libsupport.a \
        $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la \
        $(top_builddir)/ompi/mca/coll/libmca_coll.la
ompi_coll_configfile_DEPENDENCIES = $(ompi_coll_configfile_LDADD)

distclean:
	rm -rf .deps .libs *.out *.log *.o *.trs $(check_PROGRAMS) Makefile

