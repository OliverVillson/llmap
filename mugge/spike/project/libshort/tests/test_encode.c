#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

static void check(uint64_t n, const char *want) {
  char buf[SHORT_CODE_MAX];
  memset(buf, 'x', sizeof buf);
  size_t len = short_encode(n, buf);
  if (strcmp(buf, want) != 0) fprintf(stderr, "short_encode(%llu) = \"%s\", want \"%s\"\n", (unsigned long long)n, buf, want);
  assert(strcmp(buf, want) == 0);
  assert(len == strlen(want));
}

int main(void) {
  check(0, "0");
  check(1, "1");
  check(10, "a");
  check(36, "A");
  check(61, "Z");
  check(62, "10");
  check(3843, "ZZ");
  check(123456789, "8m0Kx");
  check(UINT64_MAX, "lYGhA16ahyf");
  puts("test_encode: ok");
  return 0;
}
