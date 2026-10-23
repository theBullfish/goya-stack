# Hardware

Everything here was measured on our own HL-100 cards unless it says otherwise.

## The card

| | |
|---|---|
| Board / ASIC | Habana HL-100 board, HL-1000 "Goya" ASIC |
| Memory | 8 GB DRAM on the card, plus ~50 MB on-chip SRAM |
| Engines | 1 MME (matrix engine) and 8 TPC (programmable vector cores) |
| PCIe | Gen4 x16, vendor:device `1da3:0001` |
| BARs | BAR0 256 MB (SRAM + config), BAR2 4 KB (MSI-X), BAR4 256 MB (DRAM window) |

## Power

- **About 100 W per card.** Ignore the 200 W figure on some spec sheets, which is wrong for this part.
- **Power connector: EPS / CPU 8-pin.** Use the 4+4 "CPU" cable from the power supply.
  **Never plug a PCIe 6+2 cable into it.** The pinout is different and it can kill the card.
- Many desktop power supplies have only one or two EPS cables and the motherboard uses them.
  You may need a supply with a spare EPS cable, or a second supply just for the card.
- If the card runs from its own supply, switch the card's supply on first and give it 30 seconds
  before powering the host. A host reboot does not power-cycle a card on a separate supply.
- Run [`deploy/`](../deploy) once. It caps each card at 100 W instead of the driver's 200 W
  default. On our cards the cap cost no measurable speed, and sustained four-card load ran with
  zero resets at 100 W.

## Cooling

The heatsink is **passive**. It is a server card and expects chassis airflow across the fins.
In a desktop case or open frame, point a fan straight through the heatsink.

## BIOS settings

- **Above 4G Decoding: ON.** Each card needs about 512 MB of address space. Without it the card
  shows up in `lspci` but its memory windows never get assigned.
- **Secure Boot: OFF.** The driver in this repo is built from source and is not signed, and the
  firmware tools need raw PCI access that kernel lockdown blocks.
- Use an x16 slot wired to the CPU if you can. An older Gen3 host runs the card at Gen3, which
  is the host's limit, not a fault.

## First check after seating the card

```bash
lspci -d 1da3: -nn
sudo lspci -d 1da3: -vv | grep -E "LnkCap|LnkSta"
```

`LnkSta` is the real link. On a good slot you should see `Speed 16GT/s, Width x16`
(or `8GT/s` on a Gen3 host). If `lspci -d 1da3:` prints nothing, see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-card-is-not-in-lspci).
