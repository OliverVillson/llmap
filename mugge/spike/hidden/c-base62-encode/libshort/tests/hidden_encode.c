/* Hidden test for short_encode: digit boundaries, powers of 62, and a parse-back invariant. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

static uint64_t parse(const char *s) {
  uint64_t n = 0;
  for (; *s; s++) {
    const char *p = strchr(SHORT_ALPHABET, *s);
    assert(p != NULL && *s != '\0');
    n = n * 62 + (uint64_t)(p - SHORT_ALPHABET);
  }
  return n;
}

static void check(uint64_t n, const char *want) {
  char buf[SHORT_CODE_MAX];
  memset(buf, 'x', sizeof buf);
  size_t len = short_encode(n, buf);
  if (strcmp(buf, want) != 0) fprintf(stderr, "short_encode(%llu) = \"%s\", want \"%s\"\n", (unsigned long long)n, buf, want);
  assert(strcmp(buf, want) == 0);
  assert(len == strlen(want));
}

static void invariant(uint64_t n) {
  char buf[SHORT_CODE_MAX];
  memset(buf, 'x', sizeof buf);
  size_t len = short_encode(n, buf);
  assert(len >= 1 && len < SHORT_CODE_MAX);
  assert(buf[len] == '\0');
  assert(strlen(buf) == len);
  if (n > 0) assert(buf[0] != '0'); /* no padding */
  if (parse(buf) != n) fprintf(stderr, "short_encode(%llu) = \"%s\" does not parse back\n", (unsigned long long)n, buf);
  assert(parse(buf) == n);
}

int main(void) {
  for (int d = 0; d < 62; d++) {
    char want[2] = {SHORT_ALPHABET[d], '\0'};
    check((uint64_t)d, want);
  }
  check(63, "11");
  check(3844, "100");
  check(238327, "ZZZ");
  check(238328, "1000");
  check(916132832ull, "100000");      /* 62^5 */
  check(916132831ull, "ZZZZZ");
  check(UINT64_MAX - 1, "lYGhA16ahye");

  uint64_t p = 1;
  for (int k = 0; k <= 10; k++) {
    invariant(p);
    invariant(p - 1);
    invariant(p + 1);
    p *= 62;
  }
  uint64_t x = 0x9e3779b97f4a7c15ull;
  for (int i = 0; i < 20000; i++) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    invariant(x);
    invariant(x >> (i % 64));
  }
  invariant(UINT64_MAX);
  puts("hidden_encode: ok");
  return 0;
}
