/*
 * Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2005 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2011-2012 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2015      Research Organization for Information Science
 *                         and Technology (RIST). All rights reserved.
 * Copyright (c) 2020      Sandia National Laboratories. All rights reserved.
 * Copyright (c) 2021-2024 BULL S.A.S. All rights reserved.
 * Copyright (c) 2023      Triad National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"

#include "ompi/mpi/fortran/mpif-h/bindings.h"
#include "ompi/mpi/fortran/base/constants.h"

#if OMPI_BUILD_MPI_PROFILING
#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak PMPI_PRECV_INIT = ompi_precv_init_f
#pragma weak pmpi_precv_init = ompi_precv_init_f
#pragma weak pmpi_precv_init_ = ompi_precv_init_f
#pragma weak pmpi_precv_init__ = ompi_precv_init_f

#pragma weak PMPI_Precv_init_f = ompi_precv_init_f
#pragma weak PMPI_Precv_init_f08 = ompi_precv_init_f
#else
OMPI_GENERATE_F77_BINDINGS (PMPI_PRECV_INIT,
                           pmpi_precv_init,
                           pmpi_precv_init_,
                           pmpi_precv_init__,
                           pompi_precv_init_f,
                           (char *buf, MPI_Fint *partitions, MPI_Count *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                           (buf, partitions, count, datatype, dest, tag, comm, info, request, ierr) )
#endif
#endif

#if OPAL_HAVE_WEAK_SYMBOLS
#pragma weak MPI_PRECV_INIT = ompi_precv_init_f
#pragma weak mpi_precv_init = ompi_precv_init_f
#pragma weak mpi_precv_init_ = ompi_precv_init_f
#pragma weak mpi_precv_init__ = ompi_precv_init_f

#pragma weak MPI_Precv_init_f = ompi_precv_init_f
#pragma weak MPI_Precv_init_f08 = ompi_precv_init_f
#else
#if ! OMPI_BUILD_MPI_PROFILING
OMPI_GENERATE_F77_BINDINGS (MPI_PRECV_INIT,
                           mpi_precv_init,
                           mpi_precv_init_,
                           mpi_precv_init__,
                           ompi_precv_init_f,
                           (char *buf, MPI_Fint *partitions, MPI_Count *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr),
                           (buf, partitions, count, datatype, dest, tag, comm, info, request, ierr) )
#else
#define ompi_precv_init_f pompi_precv_init_f
#endif
#endif


void ompi_precv_init_f(char *buf, MPI_Fint *partitions, MPI_Count *count, MPI_Fint *datatype, MPI_Fint *dest, MPI_Fint *tag, MPI_Fint *comm, MPI_Fint *info, MPI_Fint *request, MPI_Fint *ierr)
{
   int c_ierr;
   MPI_Info c_info;
   MPI_Datatype c_type = PMPI_Type_f2c(*datatype);
   MPI_Request c_req;
   MPI_Comm c_comm;

   c_info = PMPI_Info_f2c(*info);
   c_comm = PMPI_Comm_f2c (*comm);

   c_ierr = PMPI_Precv_init(OMPI_F2C_BOTTOM(buf), 
                      OMPI_FINT_2_INT(*partitions),
                      *count,
                      c_type, OMPI_FINT_2_INT(*dest),
                      OMPI_FINT_2_INT(*tag),
                      c_comm, c_info, &c_req);
   if (NULL != ierr) *ierr = OMPI_INT_2_FINT(c_ierr);

   if (MPI_SUCCESS == c_ierr) {
      *request = PMPI_Request_c2f(c_req);
   }
}
