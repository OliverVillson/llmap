#include <assert.h>
#include <stdio.h>
#include <string.h>
#include "short.h"

int main(void) {
  short_table *t = table_new(4); /* few buckets, so chains get long */
  assert(t != NULL);
  assert(table_len(t) == 0);
  assert(table_get(t, "missing") == NULL);

  char key[32], val[32];
  for (int i = 0; i < 200; i++) {
    snprintf(key, sizeof key, "key-%d", i);
    snprintf(val, sizeof val, "val-%d", i);
    assert(table_set(t, key, val) == 0);
  }
  strcpy(key, "clobbered"); /* the table must have copied the strings */
  strcpy(val, "clobbered");
  assert(table_len(t) == 200);
  for (int i = 0; i < 200; i++) {
    char k[32], v[32];
    snprintf(k, sizeof k, "key-%d", i);
    snprintf(v, sizeof v, "val-%d", i);
    const char *got = table_get(t, k);
    assert(got != NULL && strcmp(got, v) == 0);
  }

  assert(table_set(t, "key-7", "seven") == 0);
  assert(strcmp(table_get(t, "key-7"), "seven") == 0);
  assert(table_len(t) == 200);
  assert(table_get(t, "key-200") == NULL);
  assert(table_set(t, NULL, "x") == -1);
  assert(table_set(t, "x", NULL) == -1);
  table_free(t);
  table_free(NULL);

  short_table *one = table_new(0);
  assert(one != NULL);
  assert(table_set(one, "a", "1") == 0 && table_set(one, "b", "2") == 0);
  assert(strcmp(table_get(one, "a"), "1") == 0 && strcmp(table_get(one, "b"), "2") == 0);
  table_free(one);
  puts("test_table: ok");
  return 0;
}
