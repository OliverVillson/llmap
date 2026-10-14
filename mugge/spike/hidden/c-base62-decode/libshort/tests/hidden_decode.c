/* Hidden test for short_decode: every digit, leading zeros, overflow edges, *out untouched on error. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

static void ok(const char *s, uint64_t want) {
  uint64_t got = 12345;
  int rc = short_decode(s, &got);
  if (rc != 0 || got != want) fprintf(stderr, "short_decode(\"%s\") = %d, %llu; want 0, %llu\n", s, rc, (unsigned long long)got, (unsigned long long)want);
  assert(rc == 0 && got == want);
}

static void bad(const char *s) {
  uint64_t got = 4242;
  int rc = short_decode(s, &got);
  if (rc != -1 || got != 4242) fprintf(stderr, "short_decode(\"%s\") = %d, out %llu; want -1, out untouched\n", s, rc, (unsigned long long)got);
  assert(rc == -1);
  assert(got == 4242);
}

static void enc(uint64_t n, char *out) { /* independent reference encoder */
  char tmp[SHORT_CODE_MAX];
  size_t len = 0;
  do {
    tmp[len++] = SHORT_ALPHABET[n % 62];
    n /= 62;
  } while (n);
  for (size_t i = 0; i < len; i++) out[i] = tmp[len - 1 - i];
  out[len] = '\0';
}

int main(void) {
  for (int d = 0; d < 62; d++) {
    char s[2] = {SHORT_ALPHABET[d], '\0'};
    ok(s, (uint64_t)d);
  }
  ok("00", 0);
  ok("01", 1);
  ok("000000000000000000000000Z", 61); /* long, but no overflow */
  ok("0lYGhA16ahyf", UINT64_MAX);
  ok("lYGhA16ahye", UINT64_MAX - 1);
  ok("100", 3844);
  ok("aZ", 10 * 62 + 61);

  bad("lYGhA16ahyZ");
  bad("lYGhA16ahz0");
  bad("lYGhA16ai00");
  bad("mYGhA16ahyf");
  bad("100000000000");   /* 62^11 overflows */
  bad("ZZZZZZZZZZZ");
  bad("zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz");
  bad("12!");
  bad("-1");
  bad("+1");
  bad(" 1");
  bad("1 ");
  bad("1\n");
  bad("_");
  bad("\xc3\xa9");
  bad("ab\x7f");
  bad("\x80");

  char buf[SHORT_CODE_MAX];
  uint64_t x = 0x243f6a8885a308d3ull;
  for (int i = 0; i < 20000; i++) {
    x = x * 6364136223846793005ull + 1442695040888963407ull;
    uint64_t n = x >> (i % 64);
    enc(n, buf);
    ok(buf, n);
  }
  puts("hidden_decode: ok");
  return 0;
}
