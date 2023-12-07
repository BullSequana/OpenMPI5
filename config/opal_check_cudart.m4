dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2004-2010 The Trustees of Indiana University and Indiana
dnl                         University Research and Technology
dnl                         Corporation.  All rights reserved.
dnl Copyright (c) 2004-2005 The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
dnl                         University of Stuttgart.  All rights reserved.
dnl Copyright (c) 2004-2005 The Regents of the University of California.
dnl                         All rights reserved.
dnl Copyright (c) 2006-2016 Cisco Systems, Inc.  All rights reserved.
dnl Copyright (c) 2007      Sun Microsystems, Inc.  All rights reserved.
dnl Copyright (c) 2009      IBM Corporation.  All rights reserved.
dnl Copyright (c) 2009      Los Alamos National Security, LLC.  All rights
dnl                         reserved.
dnl Copyright (c) 2009-2011 Oak Ridge National Labs.  All rights reserved.
dnl Copyright (c) 2011-2015 NVIDIA Corporation.  All rights reserved.
dnl Copyright (c) 2015      Research Organization for Information Science
dnl                         and Technology (RIST). All rights reserved.
dnl
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

AC_DEFUN([OPAL_CHECK_CUDART],[
#
# Check to see if user wants CUDART support
#
AC_ARG_WITH([cudart],
            [AC_HELP_STRING([--with-cudart(=DIR)],
            [Build cudart support, optionally adding DIR/include])])
AC_MSG_CHECKING([if --with-cudart is set])

# Note that CUDART support is off by default.  To turn it on, the user has to
# request it.  The user can just ask for --with-cudart and it that case we
# look for the cuda_runtime_api.h file in /usr/local/cuda.  Otherwise, they can give
# us a directory.  If they provide a directory, we will look in that directory
# as well as the directory with the "include" string appended to it.  The fact
# that we check in two directories precludes us from using the OMPI_CHECK_DIR
# macro as that would error out after not finding it in the first directory.
# Note that anywhere CUDART aware code is in the Open MPI repository requires
# us to make use of AC_REQUIRE to ensure this check has been done.
AS_IF([test "$with_cudart" = "no" || test "x$with_cudart" = "x"],
      [opal_check_cudart_happy="no"
       AC_MSG_RESULT([not set (--with-cudart=$with_cudart)])],
      [AS_IF([test "$with_cudart" = "yes"],
             [AS_IF([test "x`ls /usr/local/cuda/include/cuda_runtime_api.h 2> /dev/null`" = "x"],
                    [AC_MSG_RESULT([not found in standard location])
                     AC_MSG_WARN([Expected file /usr/local/cuda/include/cuda_runtime_api.h not found])
                     AC_MSG_ERROR([Cannot continue])],
                    [AC_MSG_RESULT([found])
                     opal_check_cudart_happy=yes
                     opal_cudart_incdir=/usr/local/cuda/include])],
             [AS_IF([test ! -d "$with_cudart"],
                    [AC_MSG_RESULT([not found])
                     AC_MSG_WARN([Directory $with_cudart not found])
                     AC_MSG_ERROR([Cannot continue])],
                    [AS_IF([test "x`ls $with_cudart/include/cuda_runtime_api.h 2> /dev/null`" = "x"],
                           [AS_IF([test "x`ls $with_cudart/cuda_runtime_api.h 2> /dev/null`" = "x"],
                                  [AC_MSG_RESULT([not found])
                                   AC_MSG_WARN([Could not find cuda_runtime_api.h in $with_cudart/include or $with_cudart])
                                   AC_MSG_ERROR([Cannot continue])],
                                  [opal_check_cudart_happy=yes
                                   opal_cudart_incdir=$with_cudart
                                   AC_MSG_RESULT([found ($with_cudart/cuda_runtime_api.h)])
                                   AC_DEFINE_UNQUOTED([OPAL_CUDART_PATH],
                                                      ["$with_cudart/lib64"],
                                                      [Additional path to look for cudart library at runtime])
                           ])],
                           [opal_check_cudart_happy=yes
                            opal_cudart_incdir="$with_cudart/include"
                            AC_MSG_RESULT([found ($opal_cudart_incdir/cuda_runtime_api.h)])
                            AC_DEFINE_UNQUOTED([OPAL_CUDART_PATH],
                                               ["$with_cudart/lib64"],
                                               [Additional path to look for cudart library at runtime])
                           ])])])])


# If we have CUDART support, check to see if we have support for cudaDeviceGetPCIBusId
AS_IF([test "$opal_check_cudart_happy"="yes"],
    AC_CHECK_DECL([cudaDeviceGetPCIBusId], [CUDA_DEV_PCI_BUS=1], [CUDA_DEV_PCI_BUS=0],
        [#include <$opal_cudart_incdir/cuda_runtime_api.h>]),
    [])

AC_MSG_CHECKING([if have cuda support])
if test "$opal_check_cudart_happy" = "yes"; then
    AC_MSG_RESULT([yes (-I$opal_cudart_incdir)])
    CUDART_SUPPORT=1
    opal_datatype_cudart_CPPFLAGS="-I$opal_cudart_incdir"
    AC_SUBST([opal_datatype_cudart_CPPFLAGS])
else
    AC_MSG_RESULT([no])
    CUDART_SUPPORT=0
fi

OPAL_SUMMARY_ADD([[Miscellaneous]],[[CUDART support]],[opal_cudart], [$opal_check_cudart_happy])

AM_CONDITIONAL([OPAL_cudart_support], [test "x$CUDART_SUPPORT" = "x1"])
AC_DEFINE_UNQUOTED([OPAL_CUDART_SUPPORT],$CUDART_SUPPORT,
                   [Whether we want cudart support])

])
