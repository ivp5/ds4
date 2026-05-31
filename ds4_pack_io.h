/* ds4_pack_io.h — shared read-only mmap open/close for codec pack readers.
 *
 * Consolidation (campaign #523 structural pass): every VQB/polar codec reader duplicated the same
 * open()→fstat()→min-size→mmap()→4-byte-magic→error-cleanup prologue (~31 lines each) and the same
 * munmap+close teardown. That duplicated cleanup is exactly where a forgotten munmap/close leaks on
 * an error path. Centralizing it here is one correct implementation instead of N drifting copies
 * (consolidation + foolproofing) without touching any reader's public API (caller-safe). */
#ifndef DS4_PACK_IO_H
#define DS4_PACK_IO_H

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* mmap PATH read-only; require map_size >= min_bytes and a leading 4-byte `magic`.
 * On success: returns the mapped base pointer and writes *map / *map_size / *fd.
 * On failure: logs "tag: ..." to stderr, releases every resource it acquired, leaves
 * *map=NULL / *map_size=0 / *fd=-1, and returns NULL. */
static inline const uint8_t *ds4_pack_mmap_open_flags(const char *path, const char *tag,
                                                      const char *magic, size_t min_bytes, int mmap_flags,
                                                      void **map, size_t *map_size, int *fd) {
    *map = NULL; *map_size = 0; *fd = -1;
    int f = open(path, O_RDONLY);
    if (f < 0) {
        fprintf(stderr, "%s: open(%s) failed: %s\n", tag, path, strerror(errno));
        return NULL;
    }
    struct stat st;
    if (fstat(f, &st) != 0) {
        fprintf(stderr, "%s: fstat(%s) failed: %s\n", tag, path, strerror(errno));
        close(f); return NULL;
    }
    if ((size_t)st.st_size < min_bytes) {
        fprintf(stderr, "%s: %s too small (%lld bytes)\n", tag, path, (long long)st.st_size);
        close(f); return NULL;
    }
    void *m = mmap(NULL, (size_t)st.st_size, PROT_READ, mmap_flags, f, 0);
    if (m == MAP_FAILED) {
        fprintf(stderr, "%s: mmap(%s) failed: %s\n", tag, path, strerror(errno));
        close(f); return NULL;
    }
    if (magic && memcmp(m, magic, 4) != 0) {   /* magic==NULL: caller verifies its own (e.g. numeric) magic */
        const uint8_t *b = (const uint8_t *)m;
        fprintf(stderr, "%s: %s bad magic (got %02x %02x %02x %02x)\n",
                tag, path, b[0], b[1], b[2], b[3]);
        munmap(m, (size_t)st.st_size); close(f); return NULL;
    }
    *map = m; *map_size = (size_t)st.st_size; *fd = f;
    return (const uint8_t *)m;
}

/* MAP_SHARED convenience wrapper (the common case used by the VQB readers). */
static inline const uint8_t *ds4_pack_mmap_open(const char *path, const char *tag,
                                                const char *magic, size_t min_bytes,
                                                void **map, size_t *map_size, int *fd) {
    return ds4_pack_mmap_open_flags(path, tag, magic, min_bytes, MAP_SHARED, map, map_size, fd);
}

/* Release a pack mapping: munmap + close, then reset the {map,map_size,fd} triple to defaults. */
static inline void ds4_pack_munmap_close(void **map, size_t *map_size, int *fd) {
    if (*map && *map != MAP_FAILED) munmap(*map, *map_size);
    if (*fd >= 0) close(*fd);
    *map = NULL; *map_size = 0; *fd = -1;
}

#endif /* DS4_PACK_IO_H */
