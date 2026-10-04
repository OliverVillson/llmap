#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

int main(void) {
  shortener *s = shortener_new();
  assert(s != NULL);
  assert(shortener_count(s) == 0);

  const char *a = shortener_add(s, "https://a.example/");
  assert(a != NULL && strcmp(a, "1") == 0);
  const char *b = shortener_add(s, "https://b.example/");
  assert(b != NULL && strcmp(b, "2") == 0);
  const char *again = shortener_add(s, "https://a.example/");
  assert(again != NULL && strcmp(again, "1") == 0);
  assert(shortener_count(s) == 2);

  char url[64];
  const char *code = NULL;
  for (int i = 3; i <= 62; i++) {
    snprintf(url, sizeof url, "https://example.com/%d", i);
    code = shortener_add(s, url);
  }
  assert(code != NULL && strcmp(code, "10") == 0); /* id 62 */
  assert(strcmp(shortener_lookup(s, "10"), "https://example.com/62") == 0);
  assert(strcmp(shortener_lookup(s, "2"), "https://b.example/") == 0);
  assert(shortener_lookup(s, "zz") == NULL);
  assert(shortener_add(s, NULL) == NULL);
  assert(shortener_count(s) == 62);
  shortener_free(s);
  shortener_free(NULL);
  puts("test_short: ok");
  return 0;
}
