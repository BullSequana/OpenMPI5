/*
 * Copyright (c) 2004-2006 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2022-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

/** @file
 *
 * Compiler-specific prefetch functions
 *
 * A small set of prefetch / prediction interfaces for using compiler
 * directives to improve memory prefetching and branch prediction
 */

#ifndef OPAL_PREFETCH_H
#define OPAL_PREFETCH_H

#if defined(c_plusplus) || defined(__cplusplus)
/* C++ code */

#    if OMPI_CXX_HAVE_BUILTIN_EXPECT
#        define OPAL_LIKELY(expression)   __builtin_expect(!!(expression), 1)
#        define OPAL_UNLIKELY(expression) __builtin_expect(!!(expression), 0)
#    else
#        define OPAL_LIKELY(expression)   (expression)
#        define OPAL_UNLIKELY(expression) (expression)
#    endif

#    if defined(OMPI_CXX_HAVE_BUILTIN_PREFETCH) && OMPI_CXX_HAVE_BUILTIN_PREFETCH
#        define OPAL_PREFETCH(address, rw, locality) __builtin_prefetch(address, rw, locality)
#    else
#        define OPAL_PREFETCH(address, rw, locality)
#    endif

#else
/* C code */

#    if defined(OPAL_C_HAVE_BUILTIN_EXPECT) && OPAL_C_HAVE_BUILTIN_EXPECT
#        define OPAL_LIKELY(expression)   __builtin_expect(!!(expression), 1)
#        define OPAL_UNLIKELY(expression) __builtin_expect(!!(expression), 0)
#    else
#        define OPAL_LIKELY(expression)   (expression)
#        define OPAL_UNLIKELY(expression) (expression)
#    endif

#    if defined(OPAL_C_HAVE_BUILTIN_PREFETCH) && OPAL_C_HAVE_BUILTIN_PREFETCH
#        define OPAL_PREFETCH(address, rw, locality) __builtin_prefetch(address, rw, locality)
#    else
#        define OPAL_PREFETCH(address, rw, locality)
#    endif

#endif

#endif
