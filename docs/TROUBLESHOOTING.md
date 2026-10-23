# Troubleshooting

## Read the kernel log first

```bash
sudo dmesg | grep -iE "habana|goya"
```

| what the log says | what it means | what to do |
|---|---|---|
| `Firmware file habanalabs/goya/goya-boot-fit.itb is not found! (error -2)` | The card is fine. The host has no firmware files. | Install the firmware package ([setup step](../README.md)) and reload the driver. |
| `boot-fit version 28.2.2-sec-0` ... `Found GOYA device with 8GB DRAM` ... `Successfully added device` | Healthy start. | Nothing. |
| `CPU boot ready status timeout` with `Device boot progress - Invalid status code 2` | The card's own boot firmware is corrupt. | Reflash that one card: `sudo hl-fw-loader -d <addr> -r`, then power the card off and on (see [SAFETY.md](SAFETY.md) rule 2). |
| `Device CPU packet timeout` then `Device heartbeat failed! PCI link is healthy` then `Going to reset device` | Something polled a live sensor and the card's management CPU did not answer in time. | The card recovers by itself. Stop whatever polled it ([SAFETY.md](SAFETY.md) rule 4). |

A card that reports `BTL version 9f7a1057` or `preboot version 28.2.2-sec-0` is alive and
talking to the host, whatever the log says after that.

## Check the card's state

```bash
for d in /sys/class/habanalabs/hl[0-9]*; do
  echo "$(basename $d): $(cat $d/status) @ $(cat $d/pci_addr)  resets soft=$(cat $d/soft_reset_cnt) hard=$(cat $d/hard_reset_cnt)"
done
```

You want `Operational`. A card that shows `Disabled` while its PCIe link is healthy can usually
be brought back without a reboot:

```bash
echo 1 | sudo tee /sys/bus/pci/devices/0000:<addr>/remove
echo 1 | sudo tee /sys/bus/pci/rescan
```

## The card is not in `lspci`

Nothing in software can reach a card that never shows up on the bus. The firmware tool, the
driver and a PCI rescan all need the device to be there first.

1. Check the EPS 8-pin cable is seated and the supply feeding the card is on.
2. Power everything off, including the card's supply if it has its own, wait 30 seconds, and
   power the card first.
3. Try the card in a different x16 slot.
4. Look at the slot's presence bit: `sudo lspci -s <root port> -vv | grep SltSta`.
   `PresDet-` means the slot sees no card at all, which points at power, seating or the slot.

## "The clocks are at 50 MHz"

That is normal. With no work running the driver parks the engines at 50 MHz, and it raises
them to 1.575 GHz when work is submitted. Do not read `*_clk_curr` to check
([SAFETY.md](SAFETY.md) rule 4).
