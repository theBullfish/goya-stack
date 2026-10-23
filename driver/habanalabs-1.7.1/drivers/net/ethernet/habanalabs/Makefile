# SPDX-License-Identifier: GPL-2.0-only

ifneq ($(PATCHLEVEL),)
# kbuild part of makefile

CONFIG_HABANA_EN := m

obj-$(CONFIG_HABANA_EN) := habanalabs_en.o

include $(src)/common/Makefile
habanalabs_en-y += $(HL_EN_COMMON_FILES)

include $(src)/gaudi/Makefile
habanalabs_en-y += $(HL_EN_GAUDI_FILES)

include $(src)/gaudi2/Makefile
habanalabs_en-y += $(HL_EN_GAUDI2_FILES)

include $(src)/../../../misc/habanalabs/common/Makefile.compat

habanalabs_en-y += ../../../misc/habanalabs/common/habanalabs_compat.o

LINUXINCLUDE := -I$(src)/../../../../include -I$(src) $(LINUXINCLUDE)

else
# normal makefile

KVERSION ?= $(shell uname -r)
KERNELDIR := /lib/modules/$(KVERSION)/build
SRC_DIR ?= $(shell pwd)
GIT_SHA ?= $(shell git --git-dir=${HABANALABS_ROOT}/.git rev-parse --short HEAD)
DEBUG_CFLAGS += -g -DDEBUG

DRV_CFLAGS_MODULE="-DHL_DRIVER_GIT_SHA=$(GIT_SHA)"

default:
	$(MAKE) CFLAGS_MODULE=$(DRV_CFLAGS_MODULE) -C $(KERNELDIR) M=$(SRC_DIR) modules

debug:
	$(MAKE) EXTRA_CFLAGS="$(DEBUG_CFLAGS)" CFLAGS_MODULE=$(DRV_CFLAGS_MODULE) -C $(KERNELDIR) M=$(SRC_DIR) modules

custom:
	$(MAKE) CFLAGS_MODULE=$(DRV_CFLAGS_MODULE) -C $(CUSTOMKERNELDIR) M=$(SRC_DIR) modules

debug_custom:
	$(MAKE) EXTRA_CFLAGS="$(DEBUG_CFLAGS)" CFLAGS_MODULE=$(DRV_CFLAGS_MODULE) -C $(CUSTOMKERNELDIR) M=$(SRC_DIR) modules

importer:
	$(MAKE) CFLAGS_MODULE="$(DRV_CFLAGS_MODULE) -D__IMPORTER" -C $(KERNELDIR) M=$(SRC_DIR) modules

custom_importer:
	$(MAKE) CFLAGS_MODULE="$(DRV_CFLAGS_MODULE) -D__IMPORTER" -C $(CUSTOMKERNELDIR) M=$(SRC_DIR) modules

clean:
	$(MAKE) -C $(KERNELDIR) M=$(SRC_DIR) clean
	rm -f Module.symvers
	rm -f *.ur-safe
	rm -f common/*.ur-safe
	rm -f gaudi/*.ur-safe
	rm -f gaudi2/*.ur-safe

endif
