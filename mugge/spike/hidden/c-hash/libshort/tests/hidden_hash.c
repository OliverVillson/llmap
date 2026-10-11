/* Hidden test for short_hash: more FNV-1a vectors, bytes >= 0x80 as unsigned, long input. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

static uint32_t ref(const char *s) {
  uint32_t h = 0x811c9dc5u;
  for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
    h ^= *p;
    h *= 0x01000193u;
  }
  return h;
}

static void check(const char *s, uint32_t want) {
  uint32_t got = short_hash(s);
  if (got != want) fprintf(stderr, "short_hash(\"%s\") = 0x%08x, want 0x%08x\n", s, (unsigned)got, (unsigned)want);
  assert(got == want);
}

int main(void) {
  check("foo", 0xa9f37ed7u);
  check("fooba", 0x39aaa18au);
  check("hello world", 0xd58b3fa7u);
  check("\xc3\xa9t\xc3\xa9", 0xffb58817u);
  check("\xff", 0x7a0b824eu);
  check("\x80\x81\xfe", 0x3a187c8cu);

  static char big[4096];
  memset(big, 'a', 1000);
  big[1000] = '\0';
  check(big, 0x1dd9658du);

  for (int i = 0; i < 4095; i++) big[i] = (char)(1 + (i * 37 + 11) % 255);
  big[4095] = '\0';
  for (int len = 4095; len >= 0; len -= 7) {
    big[len] = '\0';
    check(big, ref(big));
  }
  char s[2] = {0, 0};
  for (int c = 1; c < 256; c++) {
    s[0] = (char)c;
    check(s, ref(s));
  }
  /* stops at the NUL */
  check("ab\0cd", ref("ab"));
  puts("hidden_hash: ok");
  return 0;
}
