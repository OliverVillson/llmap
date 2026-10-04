#include <assert.h>
#include <stdio.h>
#include "short.h"

static void ok(const char *s, uint64_t want) {
  uint64_t got = 0;
  int rc = short_decode(s, &got);
  if (rc != 0 || got != want) fprintf(stderr, "short_decode(\"%s\") = %d, %llu; want 0, %llu\n", s, rc, (unsigned long long)got, (unsigned long long)want);
  assert(rc == 0 && got == want);
}

static void bad(const char *s) {
  uint64_t got = 42;
  int rc = short_decode(s, &got);
  if (rc != -1) fprintf(stderr, "short_decode(\"%s\") = %d, want -1\n", s ? s : "(null)", rc);
  assert(rc == -1);
  assert(got == 42);
}

int main(void) {
  ok("0", 0);
  ok("1", 1);
  ok("a", 10);
  ok("A", 36);
  ok("Z", 61);
  ok("10", 62);
  ok("ZZ", 3843);
  ok("8m0Kx", 123456789);
  ok("lYGhA16ahyf", UINT64_MAX);
  bad(NULL);
  bad("");
  bad("ab-c");
  bad("a b");
  bad("lYGhA16ahyg");
  bad("zzzzzzzzzzzz");
  puts("test_decode: ok");
  return 0;
}
