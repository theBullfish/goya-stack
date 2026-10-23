# Rules that keep the card alive

Each rule here comes from a card we lost, a host we hung, or a reset we caused.

1. **EPS 8-pin power only.** Never a PCIe 6+2 cable. See [HARDWARE.md](HARDWARE.md#power).

2. **Never run `hl-fw-loader -r` without `-d <pci address>`.** With no `-d` it reprograms every
   Goya in the machine at once. Always name the one card, for example
   `sudo hl-fw-loader -d 03:00.0 -r`, and only for a card that actually needs it
   (see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)).

3. **Never use a CH341A or any 3.3 V SPI programmer on the card's flash chip.** The chip is a
   1.8 V part (MX25U series). A CH341A destroyed one of our cards.

4. **Do not poll live sensors.** Some driver files ask the card's management CPU for a live
   reading, and on these cards that query can time out. When it does, the driver resets the
   card and any running work is lost. Do not run `hl-smi` in a loop, and do not poll
   `*_clk_curr` or the hwmon temperature, current or power inputs.
   Static files like `status`, `pci_addr`, `fuse_ver`, `soft_reset_cnt` and `hard_reset_cnt`
   are safe to read as often as you like.

5. **One process owns a card at a time.** A second process that tries to open the same card
   gets an error. That is the driver enforcing ownership, not a broken card.

6. **Do not reload the driver on a machine with several cards while they are in use.**
   `rmmod`/`modprobe` resets every card at once.

7. **Do not write to the card's registers directly** (BAR0 pokes, `setpci` on reset bits,
   secondary bus reset, FLR, forcing D3). Several of the card's registers reset or brick it,
   and even some reads are unsafe. Everything in this repo goes through the driver.
