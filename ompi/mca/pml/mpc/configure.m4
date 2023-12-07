# -*- shell-script -*-
#
# Copyright (c)
# Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# MCA_ompi_pml_mpc_POST_CONFIG(will_build)
# to use if the MPC PML requires a endpoint tag to compile

# MCA_ompi_pml_mpc_CONFIG(action-if-can-compile,
#                        [action-if-cant-compile])
# We can always build, unless we were explicitly disabled.
AC_DEFUN([MCA_ompi_pml_mpc_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/pml/mpc/Makefile])

    # Check for MPC Lowcomm
    OPAL_CHECK_MPC_LOWCOMM([pml_mpc])

    AS_IF([test "$opal_mpc_lowcomm_happy" = "yes"],
          [$1],
          [$2])

])dnl

