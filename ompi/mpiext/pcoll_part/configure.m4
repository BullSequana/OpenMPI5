# -*- shell-script -*-
#
# Copyright (c) 2017-2018 FUJITSU LIMITED.  All rights reserved.
# Copyright (c) 2018      Research Organization for Information Science
#                         and Technology (RIST). All rights reserved.
# Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# OMPI_MPIEXT_pcoll_part_CONFIG([action-if-found], [action-if-not-found])
# -----------------------------------------------------------
AC_DEFUN([OMPI_MPIEXT_pcoll_part_CONFIG],[
    AC_CONFIG_FILES([
        ompi/mpiext/pcoll_part/Makefile
        ompi/mpiext/pcoll_part/c/Makefile
    ])

    AS_IF([test "$OMPI_WANT_MPI_PARTITIONED_COLL_SUPPORT" = "1" || \
           test "$ENABLE_EXT_ALL" = "1"],
          [$1],
          [$2])
])
