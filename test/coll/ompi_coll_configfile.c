/*
 * Copyright (c) 2019-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <dirent.h>

#include "support.h"
#include "ompi/mca/coll/base/coll_base_functions.h"
#include "ompi/mca/coll/base/coll_base_dynamic_rules.h"
#include "ompi/mca/coll/base/coll_base_dynamic_file.h"

#define EXTENT ".rules"

#define STRINGOUT(x) #x
#define TOSTRING(x) STRINGOUT(x)

static int test_config(char* file, int format);

int main(int argc, char* argv[])
{
    test_init("ompi_coll_config_file");
    #ifndef RULES_FILE_DIR
        /* Path unknown : can't parse files */
        char msg[65];
        sprintf(msg, "Collective rules file can't be found\n");
        test_failure(msg);
    #else
        char* directory=(char*)TOSTRING(RULES_FILE_DIR);
        DIR *dir = opendir(directory);

        if (dir == NULL) {
            /* Open failed : can't parse files */
            char msg[65];
            sprintf(msg, "Collective rules file can't be found\n");
            test_failure(msg);
        }
        else {
            int failed=0;
            int succeeded=0;
            struct dirent *de;

            while ((de = readdir(dir)) != NULL) {
                /* for each file in directory */
                char *extent_search = strstr(de->d_name, EXTENT);
                if(extent_search && !strcmp(extent_search, EXTENT)) {
                    char rules_file_path[256];
                    sprintf(rules_file_path,"%s/%s",directory,de->d_name);
                    /* Contains ".rules" at the end : it's a config file */
                    if(test_config(rules_file_path, 2) < 0
                       && test_config(rules_file_path, 1) < 0
                       && test_config(rules_file_path, 0) < 0) {
                        char msg[65];
                        /* Parser returned error : something wrong with this file */
                        sprintf(msg, "base config file %s parsing failed\n", de->d_name);
                        test_failure(msg);
                        failed++;
                    }
                    else {
                        succeeded++;
                    }
                }
            }
            if(!failed) {
                /* No error returned */
                if(succeeded) {
                    /* At least one success (that means one file parsed) */
                    test_success();
                }
                else {
                    /* No file parsed : probably a bug... */
                    char msg[65];
                    sprintf(msg, "No config file were parsed\n");
                    test_failure(msg);
                }
            }
            closedir(dir);
        }
    #endif
    /* All done */
    return test_finalize();
}

int test_config(char* file, int format)
{
    int rc;
    ompi_coll_base_alg_rule_t* rules = NULL;
    rc = ompi_coll_base_read_rules_config_file( file,
                                                format,
                                                &rules,
                                                COLLCOUNT);
    if(rc >= 0) {
        rc = ompi_coll_base_dump_all_rules (rules, COLLCOUNT);
        if(rc >= 0) {
            rc = ompi_coll_base_free_all_rules (rules, COLLCOUNT);
        }
    }
    return rc;
}
