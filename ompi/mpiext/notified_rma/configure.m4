# -*- shell-script -*-
#
# Copyright (c) 2004-2009 The Trustees of Indiana University.
#                         All rights reserved.
# Copyright (c) 2012-2015 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# OMPI_MPIEXT_notified_rma_CONFIG([action-if-found], [action-if-not-found])
# -----------------------------------------------------------
AC_DEFUN([OMPI_MPIEXT_notified_rma_CONFIG],[
    AC_CONFIG_FILES([ompi/mpiext/notified_rma/Makefile])
    AC_CONFIG_FILES([ompi/mpiext/notified_rma/c/Makefile])
    AC_CONFIG_HEADER([ompi/mpiext/notified_rma/c/mpiext_notified_rma_c.h])
    AC_CONFIG_FILES([ompi/mpiext/notified_rma/mpif-h/Makefile])
    AC_CONFIG_FILES([ompi/mpiext/notified_rma/use-mpi-f08/Makefile])
    AC_CONFIG_FILES([ompi/mpiext/notified_rma/use-mpi/Makefile])

    # Build notified rma only if it is activated by configure options
    dnl explicit extension deactivation take the priority
    AS_IF([test "$OMPI_WANT_MPI_NOTIFS_SUPPORT" = "0"],
          [AC_MSG_RESULT([notified RMA explicitly disabled])
          [$2]],

          dnl Check then if this extension is in the list of mpi extensions or if all are requested
          [AS_IF([test "$ENABLE_EXT_ALL" = "1" || \
                 test "$ENABLE_notified_rma" = "1"],
                 [AC_MSG_RESULT([Asked for a list of mpiext including notifed RMA])
                 [$1]],
                 [$2])
          ])
])

