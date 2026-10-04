CC ?= cc
CFLAGS = -Wall -Werror -std=c11 -O1 -Iinclude
B = build
SRC = src/encode.c src/decode.c src/hash.c src/table.c src/shortener.c

.PHONY: lib test test-encode test-decode test-hash test-table test-short clean

lib: $(B)/libshort.so

$(B)/libshort.so: $(SRC) include/short.h
	@mkdir -p $(B)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ $(SRC)

define run_test
	@mkdir -p $(B)
	$(CC) $(CFLAGS) -o $(B)/$(1) tests/$(1).c $(2)
	./$(B)/$(1)
endef

test-encode:
	$(call run_test,test_encode,src/encode.c)
test-decode:
	$(call run_test,test_decode,src/decode.c)
test-hash:
	$(call run_test,test_hash,src/hash.c)
test-table:
	$(call run_test,test_table,src/table.c src/hash.c)
test-short:
	$(call run_test,test_short,src/shortener.c src/table.c src/hash.c src/encode.c)

test: test-encode test-decode test-hash test-table test-short

clean:
	rm -rf $(B)
