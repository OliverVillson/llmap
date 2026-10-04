#include <assert.h>
#include <stdio.h>
#include "short.h"

static void check(const char *s, uint32_t want) {
  uint32_t got = short_hash(s);
  if (got != want) fprintf(stderr, "short_hash(\"%s\") = 0x%08x, want 0x%08x\n", s, (unsigned)got, (unsigned)want);
  assert(got == want);
}

int main(void) {
  check("", 0x811c9dc5u);
  check("a", 0xe40c292cu);
  check("foobar", 0xbf9cf968u);
  assert(short_hash("abc") != short_hash("acb"));
  puts("test_hash: ok");
  return 0;
}
