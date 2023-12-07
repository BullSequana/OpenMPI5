dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

dnl
dnl OPAL_CHECK_MPC_LOWCOMM
dnl --------------------------------------------------------
dnl Check for MPC Lowcomm networking library.
dnl Upon return:
dnl
dnl - opal_mpc_lowcomm_happy: will be "yes" or "no"
dnl - opal_mpc_lowcomm_{CPPFLAGS|LDFLAGS|LIBS} will be loaded (if relevant)
dnl
AC_DEFUN([OPAL_CHECK_MPC_LOWCOMM],[
    # Add --with options
    AC_ARG_WITH([mpc-lowcomm],
                [AC_HELP_STRING([--with-mpc-lowcomm=DIR],
                                [Specify location of MPC Lowcomm (experimental support). Error if support cannot be found.])])

    AC_ARG_WITH([mpc-lowcomm-libdir],
                [AC_HELP_STRING([--with-mpc-lowcomm-libdir=DIR],
                                [Search for MPC Lowcomm libraries in DIR])])


    # Sanity check the --with values
    OPAL_CHECK_WITHDIR([mpc-lowcomm], [$with_mpc_lowcomm],
                       [include/mpcframework/mpc_lowcomm.h])
    OPAL_CHECK_WITHDIR([mpc-lowcomm-libdir], [$with_mpc_lowcomm_libdir],
                       [libmpclowcomm.so])

    OPAL_VAR_SCOPE_PUSH([opal_check_mpc_lowcomm_save_CPPFLAGS opal_check_mpc_lowcomm_save_LDFLAGS opal_check_mpc_lowcomm_save_LIBS])
    opal_check_mpc_lowcomm_save_CPPFLAGS=$CPPFLAGS
    opal_check_mpc_lowcomm_save_LDFLAGS=$LDFLAGS
    opal_check_mpc_lowcomm_save_LIBS=$LIBS

    opal_mpc_lowcomm_happy=yes
    AS_IF([test "$with_mpc_lowcomm" = "no"],
          [opal_ofi_mpc_lowcomm=no])

    AS_IF([test $opal_mpc_lowcomm_happy = yes],
          [AC_MSG_CHECKING([looking for MPC Lowcomm in])
           AS_IF([test "$with_mpc_lowcomm" != "yes"],
                 [opal_mpc_lowcomm_dir=$with_mpc_lowcomm
                  AC_MSG_RESULT([($opal_mpc_lowcomm_dir)])],
                 [AC_MSG_RESULT([(default search paths)])])
           AS_IF([test ! -z "$with_mpc_lowcomm_libdir" && \
                         test "$with_mpc_lowcomm_libdir" != "yes"],
                 [opal_mpc_lowcomm_libdir=$with_mpc_lowcomm_libdir])
          ])


    # add non-standard include path
    # TODO: remove once MPC uses include/
    CPPFLAGS="$CPPFLAGS -I$opal_mpc_lowcomm_dir/include/mpcframework"
    opal_mpc_lowcomm_CPPFLAGS="$opal_mpc_lowcomm_CPPFLAGS -I$opal_mpc_lowcomm_dir/include/mpcframework"

    AS_IF([test $opal_mpc_lowcomm_happy = yes],
          [OAC_CHECK_PACKAGE([mpc_lowcomm],
                             [opal_mpc_lowcomm],
                             [mpc_lowcomm.h],
                             [mpclowcomm mpcconfig],
                             [mpc_lowcomm_test],
                             [],
                             [opal_mpc_lowcomm_happy=no])])

    CPPFLAGS=$opal_check_mpc_lowcomm_save_CPPFLAGS
    LDFLAGS=$opal_check_mpc_lowcomm_save_LDFLAGS
    LIBS=$opal_check_mpc_lowcomm_save_LIBS

    AC_SUBST([opal_mpc_lowcomm_CPPFLAGS])
    AC_SUBST([opal_mpc_lowcomm_LDFLAGS])
    AC_SUBST([opal_mpc_lowcomm_LIBS])

    OPAL_SUMMARY_ADD([[Transports]],[[MPC Lowcomm]],[$1],[$opal_mpc_lowcomm_happy])

    OPAL_VAR_SCOPE_POP

    AS_IF([test $opal_mpc_lowcomm_happy = no],
          [AS_IF([test -n "$with_mpc_lowcomm" && test "$with_mpc_lowcomm" != "no"],
                 [AC_MSG_WARN([MPC Lowcomm support requested (via --with-mpc-lowcomm or --with-mpc-lowcomm-libdir), BUT not found.])
                  AC_MSG_ERROR([Cannot continue.])])
           ])
])dnl
