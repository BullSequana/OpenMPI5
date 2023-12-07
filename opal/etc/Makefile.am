#
# Copyright (c) 2004-2005 The Trustees of Indiana University and Indiana
#                         University Research and Technology
#                         Corporation.  All rights reserved.
# Copyright (c) 2004-2005 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
#                         University of Stuttgart.  All rights reserved.
# Copyright (c) 2004-2005 The Regents of the University of California.
#                         All rights reserved.
# Copyright (c) 2008-2020 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2020-2024 BULL S.A.S. All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

opal_config_files = @OPAL_DEFAULT_MCA_PARAM_CONF@
opal_file_from_platform = @OPAL_PARAM_FROM_PLATFORM@

EXTRA_DIST = $(opal_config_files)

# Steal a little trickery from a generated Makefile to only install
# files if they do not already exist at the target.  Be sure to read
# thread starting here
# (https://www.open-mpi.org/community/lists/devel/2008/06/4080.php) for
# details why the mkdir is in install-data-local.

install-data-local:
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)
	$(MKDIR_P) $(DESTDIR)$(sysconfdir)/profile/
	@ p="$(opal_config_files)"; \
        if test "$(opal_file_from_platform)" = "yes"; then \
	    if test -f "$$p"; then d=; else d="$(srcdir)/"; fi; \
	    echo " $(INSTALL_DATA) $$d$$p $(DESTDIR)$(sysconfdir)/openmpi-mca-params.conf"; \
	    $(INSTALL_DATA) $$d$$p $(DESTDIR)$(sysconfdir)/openmpi-mca-params.conf; \
        profile_dir="`dirname  $$d$$p`";\
        srcpath="`readlink -f $(abs_top_srcdir)`";\
        built_profile_dir=`echo $$profile_dir | sed "s|$$srcpath|$(abs_top_builddir)|g"`;\
        profile_files="`find $$profile_dir/profile/ | grep -v "\.in$$"`" ;\
        for file in $$profile_files ; do \
            echo $(INSTALL_DATA) $$file $(DESTDIR)$(sysconfdir)/profile/ ;\
            $(INSTALL_DATA) $$file $(DESTDIR)$(sysconfdir)/profile/ ;\
        done;\
        if test "$$built_profile_dir" != "$$profile_dir" ; then \
            built_profile_files="`find $$built_profile_dir/profile/`";\
            for file in $$built_profile_files ; do \
                echo $(INSTALL_DATA) $$file $(DESTDIR)$(sysconfdir)/profile/ ;\
                $(INSTALL_DATA) $$file $(DESTDIR)$(sysconfdir)/profile/ ;\
            done;\
        fi;\
        else \
            for file in $$p; do \
	        if test -f $(DESTDIR)$(sysconfdir)/openmpi-mca-params.conf; then \
	            echo "******************************* WARNING ************************************"; \
	            echo "*** Not installing new $$file over existing file in:"; \
	            echo "***   $(DESTDIR)$(sysconfdir)/$$file"; \
	            echo "******************************* WARNING ************************************"; \
	        else \
	            if test -f "$$file"; then d=; else d="$(srcdir)/"; fi; \
	            f="`echo $$file | sed -e 's|^.*/||'`"; \
	            echo " $(INSTALL_DATA) $$d$$file $(DESTDIR)$(sysconfdir)/$$f"; \
	            $(INSTALL_DATA) $$d$$file $(DESTDIR)$(sysconfdir)/$$f; \
	        fi; \
	    done \
        fi;


# Only remove if exactly the same as what in our tree
# NOTE TO READER: Bourne shell if ... fi evaluates the body if
#    the return of the evaluated command is 0 (as opposed to non-zero
#    as used by everyone else)
uninstall-local:
	@ p="$(opal_config_files)"; \
	for file in $$p; do \
	  if test -f "$(DESTDIR)$(sysconfdir)/$$file"; then \
	  	if test -f "$$file"; then d=; else d="$(srcdir)/"; fi; \
	    if diff "$(DESTDIR)$(sysconfdir)/$$file" "$$d$$file" > /dev/null 2>&1 ; then \
	      echo "rm -f $(DESTDIR)$(sysconfdir)/$$file" ; \
	      rm -f "$(DESTDIR)$(sysconfdir)/$$file" ; \
	    fi ; \
	  fi ; \
	done
