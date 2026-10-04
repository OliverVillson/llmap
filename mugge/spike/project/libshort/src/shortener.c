#include "short.h"

/* STUB: ticket c-short fills this in. */
struct shortener {
  uint64_t next_id;
};

shortener *shortener_new(void) { return NULL; }

void shortener_free(shortener *s) {}

const char *shortener_add(shortener *s, const char *url) { return NULL; }

const char *shortener_lookup(const shortener *s, const char *code) { return NULL; }

size_t shortener_count(const shortener *s) { return 0; }
