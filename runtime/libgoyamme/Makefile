# libgoyamme: int8 GEMM on the Goya MME and kernels on the 8 TPC cores, all through the driver.
# Built by scripts/build-runtime.sh, which builds ../hl-thunk first.
HLT    ?= $(abspath ../hl-thunk)
UAPI   ?= $(abspath ../../driver/habanalabs-1.7.1/include/uapi)
PREFIX ?= /usr/local
LIBDIR ?= $(PREFIX)/lib/goya-stack
TPCDIR ?= $(PREFIX)/share/goya-stack/tpc-kernels

CFLAGS ?= -O2
CFLAGS += -Wall -Wextra -fPIC -I. -I$(UAPI) -I$(HLT)/include -I$(HLT)/include/uapi \
          -I$(HLT)/src/klib -I$(HLT)/include/specs/common -I$(HLT)/include/specs \
          -I$(HLT)/tests -I$(HLT)/tests/common \
          -DGOYA_TPC_DIR_DEFAULT='"$(TPCDIR)"'
# rpath pins our own hl-thunk 1.7.0 build, so an SDK copy of libhl-thunk.so.1 can never be picked up.
LDFLAGS += -L$(HLT)/build/lib -Wl,-rpath,$(LIBDIR)
LDLIBS   = $(HLT)/build/lib/libhl-thunk-tests-static.a -lhl-thunk -lcmocka -lpthread

SRC = goya_mme.c goya_tpc.c
HDR = goya_mme.h goya_mme_internal.h

all: libgoyamme.so goya_gemm

libgoyamme.so: $(SRC) $(HDR)
	$(CC) $(CFLAGS) -shared -o $@ $(SRC) $(LDFLAGS) $(LDLIBS)

goya_gemm: ../../examples/goya_gemm.c $(SRC) $(HDR)
	$(CC) $(CFLAGS) -o $@ ../../examples/goya_gemm.c $(SRC) $(LDFLAGS) $(LDLIBS)

install: all
	install -d $(DESTDIR)$(LIBDIR) $(DESTDIR)$(TPCDIR) $(DESTDIR)$(PREFIX)/bin $(DESTDIR)$(PREFIX)/include
	install -m 0755 libgoyamme.so $(DESTDIR)$(LIBDIR)/
	cp -P $(HLT)/build/lib/libhl-thunk.so* $(DESTDIR)$(LIBDIR)/
	install -m 0644 ../tpc-kernels/*.o $(DESTDIR)$(TPCDIR)/
	install -m 0755 goya_gemm $(DESTDIR)$(PREFIX)/bin/
	install -m 0644 goya_mme.h $(DESTDIR)$(PREFIX)/include/

clean:
	rm -f libgoyamme.so goya_gemm

.PHONY: all install clean
