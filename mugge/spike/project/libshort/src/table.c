#include "short.h"

/* STUB: ticket c-table fills this in. */
struct short_table {
  size_t len;
};

short_table *table_new(size_t buckets) { return NULL; }

void table_free(short_table *t) {}

int table_set(short_table *t, const char *key, const char *val) { return -1; }

const char *table_get(const short_table *t, const char *key) { return NULL; }

size_t table_len(const short_table *t) { return 0; }
