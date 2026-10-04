#ifndef SHORT_H
#define SHORT_H

#include <stddef.h>
#include <stdint.h>

/* Base62 digits, in value order: '0' is 0, 'a' is 10, 'A' is 36, 'Z' is 61. */
#define SHORT_ALPHABET "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

/* Longest code (UINT64_MAX is "lYGhA16ahyf", 11 digits) plus the NUL. */
#define SHORT_CODE_MAX 12

/* src/encode.c
 * Writes n in base62 (most significant digit first, no padding; 0 is "0") to out, which has
 * room for SHORT_CODE_MAX bytes, NUL-terminates it and returns its length. */
size_t short_encode(uint64_t n, char *out);

/* src/decode.c
 * Parses a base62 code into *out. Returns 0 on success, or -1 (leaving *out alone) when s is
 * NULL or empty, holds a character outside SHORT_ALPHABET, or the value overflows uint64_t. */
int short_decode(const char *s, uint64_t *out);

/* src/hash.c
 * 32-bit FNV-1a of the NUL-terminated string s: start at 2166136261 (0x811c9dc5), then for each
 * byte xor it in and multiply by 16777619 (0x01000193), modulo 2^32. */
uint32_t short_hash(const char *s);

/* src/table.c
 * A string -> string hash table with separate chaining, bucketed by short_hash(key) % buckets.
 * Keys and values are copied in; returned pointers stay valid until the key is overwritten or
 * the table is freed. */
typedef struct short_table short_table;

/* New empty table with this many buckets (0 is treated as 1). NULL if out of memory. */
short_table *table_new(size_t buckets);
/* Frees the table and every copied key and value. NULL is a no-op. */
void table_free(short_table *t);
/* Inserts or overwrites key. Returns 0 on success, -1 on NULL arguments or out of memory. */
int table_set(short_table *t, const char *key, const char *val);
/* The value stored for key, or NULL when absent. */
const char *table_get(const short_table *t, const char *key);
/* Number of distinct keys stored. */
size_t table_len(const short_table *t);

/* src/shortener.c
 * Gives each new URL the next id (starting at 1) and the code short_encode(id). Adding a URL
 * that is already stored returns its existing code. Uses two short_tables: code -> url and
 * url -> code. */
typedef struct shortener shortener;

/* New empty shortener, or NULL if out of memory. */
shortener *shortener_new(void);
/* Frees everything. NULL is a no-op. */
void shortener_free(shortener *s);
/* The code for url (owned by the shortener), or NULL on NULL arguments or out of memory. */
const char *shortener_add(shortener *s, const char *url);
/* The url stored under code (owned by the shortener), or NULL when unknown. */
const char *shortener_lookup(const shortener *s, const char *code);
/* Number of distinct URLs stored. */
size_t shortener_count(const shortener *s);

#endif
