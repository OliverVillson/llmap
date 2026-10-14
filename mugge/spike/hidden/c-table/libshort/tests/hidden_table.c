/* Hidden test for short_table: one-bucket chains, prefix and empty keys, repeated overwrites,
 * pointer stability, independent tables. */
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

static void expect(const short_table *t, const char *k, const char *v) {
  const char *got = table_get(t, k);
  if (v == NULL) {
    if (got != NULL) fprintf(stderr, "table_get(\"%s\") = \"%s\", want NULL\n", k, got);
    assert(got == NULL);
    return;
  }
  if (got == NULL || strcmp(got, v) != 0) fprintf(stderr, "table_get(\"%s\") = \"%s\", want \"%s\"\n", k, got ? got : "(null)", v);
  assert(got != NULL && strcmp(got, v) == 0);
}

int main(void) {
  /* one bucket: everything chains */
  short_table *t = table_new(1);
  assert(t != NULL);
  assert(table_set(t, "a", "1") == 0);
  assert(table_set(t, "ab", "2") == 0);
  assert(table_set(t, "abc", "3") == 0);
  assert(table_set(t, "", "empty-key") == 0);
  assert(table_set(t, "empty-val", "") == 0);
  assert(table_len(t) == 5);
  expect(t, "a", "1");
  expect(t, "ab", "2");
  expect(t, "abc", "3");
  expect(t, "", "empty-key");
  expect(t, "empty-val", "");
  expect(t, "abcd", NULL);
  expect(t, "A", NULL);

  /* pointers to other keys stay valid while one key is overwritten many times */
  const char *stable = table_get(t, "ab");
  char v[32];
  for (int i = 0; i < 500; i++) {
    snprintf(v, sizeof v, "v%d", i);
    assert(table_set(t, "a", v) == 0);
  }
  expect(t, "a", "v499");
  assert(table_len(t) == 5);
  assert(strcmp(stable, "2") == 0);

  /* a longer value replaces a shorter one and vice versa */
  assert(table_set(t, "abc", "a much longer value than before, to force a new allocation") == 0);
  expect(t, "abc", "a much longer value than before, to force a new allocation");
  assert(table_set(t, "abc", "s") == 0);
  expect(t, "abc", "s");

  /* NULL arguments */
  assert(table_set(NULL, "k", "v") == -1);
  assert(table_set(t, NULL, NULL) == -1);
  assert(table_len(t) == 5);

  /* many keys in one bucket and in many buckets */
  short_table *u = table_new(1009);
  assert(u != NULL);
  char k[32];
  for (int i = 0; i < 3000; i++) {
    snprintf(k, sizeof k, "k%d", i);
    snprintf(v, sizeof v, "%d", i * 3);
    assert(table_set(t, k, v) == 0);
    assert(table_set(u, k, v) == 0);
  }
  for (int i = 0; i < 3000; i += 2) {
    snprintf(k, sizeof k, "k%d", i);
    snprintf(v, sizeof v, "even%d", i);
    assert(table_set(u, k, v) == 0);
  }
  assert(table_len(t) == 3005);
  assert(table_len(u) == 3000);
  for (int i = 0; i < 3000; i++) {
    snprintf(k, sizeof k, "k%d", i);
    snprintf(v, sizeof v, "%d", i * 3);
    expect(t, k, v);
    if (i % 2 == 0) snprintf(v, sizeof v, "even%d", i);
    expect(u, k, v);
  }
  expect(u, "a", NULL);
  expect(u, "k3000", NULL);
  expect(t, "", "empty-key");
  table_free(t);
  table_free(u);

  short_table *e = table_new(16);
  assert(e != NULL && table_len(e) == 0);
  expect(e, "", NULL);
  table_free(e);
  puts("hidden_table: ok");
  return 0;
}
