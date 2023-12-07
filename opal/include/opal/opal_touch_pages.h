/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2023-2024 BULL S.A.S. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#include "opal/sys/atomic.h"

//Page prefetch
static inline void
opal_touch_pages_write(char * buffer, size_t len)
{
    if (NULL == buffer || sizeof(int32_t) >= len) {
        /* Cannot do atomic touch on this buffer */
        return;
    }

    /* Touch pages with atomic add.
     * To make sure writes to be aligned to 64b write pages first bytes.
     * */
    size_t page_size = sysconf(_SC_PAGESIZE);
    unsigned int page_mask = page_size - 1;

    /* In case the buffer do not start on the page first byte */
    size_t start_offset = ((uint64_t) buffer) & page_mask;
    if (0 != start_offset) {
        opal_atomic_fetch_add_32((int32_t *) buffer, 0);
    }
    buffer += start_offset;
    len -= start_offset;

    /* Touch all pages fully used by the buffer */
    size_t end_off = len & page_mask;
    const char *end = buffer + len - end_off;
    while (buffer < end) {
        opal_atomic_fetch_add_32((int32_t *) buffer, 0);
        buffer += page_size;
    }

    /* Touch last page which is not fully occupied by the buffer
     * if such a page  exists */
    if (end_off) {
        if (OPAL_UNLIKELY(end_off < sizeof(int32_t))) {
            buffer -= (sizeof(int32_t) - end_off);
        }
        opal_atomic_fetch_add_32((int32_t *) buffer, 0);
    }
    return;
}
