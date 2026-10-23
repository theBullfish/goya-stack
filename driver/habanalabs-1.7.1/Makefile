# SPDX-License-Identifier: GPL-2.0-only

HABANADIR := drivers/misc/habanalabs
HABANAENDIR := drivers/net/ethernet/habanalabs

MISC_CFLAGS=""
BUILD_IB=

# must build the Ethernet and the InfiniBand drivers first as the compute driver
# depends on them (in case aux bus is unavailable) and hence needs their symbols

default:
	$(MAKE) -C $(HABANAENDIR)
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS)
debug:
	$(MAKE) -C $(HABANAENDIR) debug
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) debug
custom:
	$(MAKE) -C $(HABANAENDIR) custom
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) custom
debug_custom:
	$(MAKE) -C $(HABANAENDIR) debug_custom
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) debug_custom
importer:
	$(MAKE) -C $(HABANAENDIR) importer
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) importer
custom_importer:
	$(MAKE) -C $(HABANAENDIR) custom_importer
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) custom_importer
clean:
	$(MAKE) -C $(HABANAENDIR) clean
	$(MAKE) -C $(HABANADIR) HL_CFLAGS=$(MISC_CFLAGS) clean

