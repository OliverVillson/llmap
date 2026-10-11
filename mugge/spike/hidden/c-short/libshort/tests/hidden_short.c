/* Hidden test for the shortener: codes are short_encode(id) well past two digits, re-adding a
 * url does not use up an id, lookups of unknown codes, returned pointers stay valid,
 * independent shorteners. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

#define N 5000

int main(void) {
  shortener *s = shortener_new();
  assert(s != NULL);
  assert(shortener_lookup(s, "1") == NULL);

  char url[64], want[SHORT_CODE_MAX];
  const char *first = shortener_add(s, "https://first.example/");
  assert(first != NULL && strcmp(first, "1") == 0);
  for (uint64_t id = 2; id <= N; id++) {
    snprintf(url, sizeof url, "https://example.com/p/%llu", (unsigned long long)id);
    const char *code = shortener_add(s, url);
    short_encode(id, want);
    if (code == NULL || strcmp(code, want) != 0) fprintf(stderr, "add #%llu = \"%s\", want \"%s\"\n", (unsigned long long)id, code ? code : "(null)", want);
    assert(code != NULL && strcmp(code, want) == 0);
    /* re-adding every 10th url gives back the same code and uses no id */
    if (id % 10 == 0) {
      const char *again = shortener_add(s, url);
      assert(again != NULL && strcmp(again, want) == 0);
    }
  }
  assert(shortener_count(s) == N);
  assert(strcmp(first, "1") == 0); /* still valid after many adds */

  for (uint64_t id = 2; id <= N; id++) {
    snprintf(url, sizeof url, "https://example.com/p/%llu", (unsigned long long)id);
    short_encode(id, want);
    const char *got = shortener_lookup(s, want);
    assert(got != NULL && strcmp(got, url) == 0);
  }
  assert(strcmp(shortener_lookup(s, "1"), "https://first.example/") == 0);
  assert(strcmp(shortener_lookup(s, "100"), "https://example.com/p/3844") == 0);
  assert(shortener_lookup(s, "0") == NULL);
  short_encode(N + 1, want);
  assert(shortener_lookup(s, want) == NULL);
  assert(shortener_lookup(s, "https://first.example/") == NULL);
  assert(shortener_lookup(s, "") == NULL);

  /* urls that differ only slightly are distinct */
  const char *x = shortener_add(s, "https://first.example");
  short_encode(N + 1, want);
  assert(x != NULL && strcmp(x, want) == 0);
  assert(shortener_count(s) == N + 1);

  /* a second shortener starts again at 1 and shares nothing */
  shortener *t = shortener_new();
  assert(t != NULL);
  assert(shortener_count(t) == 0);
  assert(shortener_lookup(t, "2") == NULL);
  const char *b = shortener_add(t, "https://example.com/p/2");
  assert(b != NULL && strcmp(b, "1") == 0);
  assert(shortener_add(NULL, "https://x.example/") == NULL);
  shortener_free(t);
  assert(shortener_count(s) == N + 1);
  shortener_free(s);
  puts("hidden_short: ok");
  return 0;
}
