/* SPDX-License-Identifier: GPL-2.0
 *
 * Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef HABANALABS_EN_H_
#define HABANALABS_EN_H_

#include <linux/habanalabs/nic.h>

#include <linux/netdevice.h>
#include <linux/pci.h>

#include "../../../../misc/habanalabs/common/habanalabs_compat.h"

#define HL_EN_NAME			"habanalabs_en"

#define HL_EN_DEBUG			0

#define HL_EN_PORT(aux_dev, idx)	\
	(&(((struct hl_en_device *)(aux_dev)->priv)->ports[(idx)]))

#define hl_netdev_priv(ndev) \
({ \
	typecheck(struct net_device *, ndev); \
	*(struct hl_en_port **)netdev_priv(ndev); \
})

/**
 * enum hl_en_eth_pkt_status - status of Rx Ethernet packet.
 * ETH_PKT_OK: packet was received successfully.
 * ETH_PKT_DROP: packet should be dropped.
 * ETH_PKT_NONE: no available packet.
 */
enum hl_en_eth_pkt_status {
	ETH_PKT_OK,
	ETH_PKT_DROP,
	ETH_PKT_NONE
};

/**
 * struct hl_en_port - manage port common structure.
 * @hdev: habanalabs Ethernet device structure.
 * @ndev: network device.
 * @wq: general purpose WQ for low/medium priority jobs like link status detection or status fetch.
 * @tx_wq: WQ for handling packet transmission outside interrupt context (for simulator only).
 * @rx_wq: WQ for Rx poll when we cannot schedule NAPI poll.
 * @mac_addr: HW MAC addresses.
 * @asic_specific: ASIC specific port structure.
 * @napi: New API structure.
 * @open_work: work for initializing the port H/W.
 * @rx_poll_work: Rx work for polling mode.
 * @in_reset: true if the NIC was marked as in reset, false otherwise. Used to avoid an additional
 *            stopping of the NIC if a hard reset was re-initiated.
 * @pflags: ethtool private flags bit mask.
 * @idx: index of this specific port.
 * @error_during_open: saved port open error for later use.
 * @is_initialized: true if the port H/W is initialized, false otherwise.
 * @pfc_enable: true if this port supports Priority Flow Control, false otherwise.
 * @auto_neg_enable: is autoneg enabled.
 * @auto_neg_resolved: was autoneg phase finished successfully.
 */
struct hl_en_port {
	struct hl_en_device	*hdev;
	struct net_device	*ndev;
	struct workqueue_struct	*wq;
	struct workqueue_struct	*tx_wq;
	struct workqueue_struct	*rx_wq;
	char			*mac_addr;
	void			*asic_specific;
	struct napi_struct	napi;
	struct delayed_work	open_work;
	struct delayed_work	rx_poll_work;
	atomic_t		in_reset;
	u32			pflags;
	u32			idx;
	s32			error_during_open;
	u8			is_initialized;
	u8			pfc_enable;
	u8			auto_neg_enable;
	u8			auto_neg_resolved;
};

/**
 * struct hl_en_asic_funcs - ASIC specific Ethernet functions.
 * @dev_init: device init.
 * @dev_fini: device cleanup.
 * @reenable_rx_irq: re-enable Rx interrupts.
 * @eth_port_open: initialize and open the Ethernet port.
 * @eth_port_close: close the Ethernet port.
 * @write_pkt_to_hw: write skb to HW.
 * @read_pkt_from_hw: read pkt from HW.
 * @get_pfc_cnts: get PFC counters.
 */
struct hl_en_asic_funcs {
	int (*dev_init)(struct hl_en_device *hdev);
	void (*dev_fini)(struct hl_en_device *hdev);
	void (*reenable_rx_irq)(struct hl_en_port *port);
	int (*eth_port_open)(struct hl_en_port *port);
	void (*eth_port_close)(struct hl_en_port *port);
	netdev_tx_t (*write_pkt_to_hw)(struct hl_en_port *port, struct sk_buff *skb);
	int (*read_pkt_from_hw)(struct hl_en_port *port, void **pkt_addr, u32 *pkt_size);
	void (*get_pfc_cnts)(struct hl_en_port *port, void *ptr);
};

/**
 * struct hl_en_device - habanalabs Ethernet device structure.
 * @pdev: pointer to PCI device, can be NULL in case of simulator device.
 * @dev: related kernel basic device structure.
 * @ports: array of all ports manage common structures.
 * @aux_dev: pointer to auxiliary device.
 * @asic_specific: ASIC specific device structure.
 * @driver_ver: Kernel driver version.
 * @fw_ver: FW version.
 * @qsfp_eeprom: QSFPD EEPROM info.
 * @mac_addr: array of all MAC addresses.
 * @asic_funcs: ASIC specific Ethernet functions.
 * @asic_type: ASIC specific type.
 * @sb_base_addr: the base address of a Tx eth pkt cyclic buffer.
 * @sb_base_size: the size of a Tx eth pkt cyclic buffer.
 * @swq_base_addr: the base address of a Tx workqueue cyclic buffer.
 * @swq_base_size: the size of a Tx workqueue cyclic buffer.
 * @ports_mask: mask of available ports.
 * @auto_neg_mask: mask of port with Autonegotiation enabled.
 * @max_num_of_ports: max number of available ports;
 * @port_reset_timeout: max time in seconds for a port reset flow to finish.
 * @pending_reset_long_timeout: long timeout for pending hard reset to finish in seconds.
 * @max_frm_len: maximum allowed frame length.
 * @raw_elem_size: size of element in raw buffers.
 * @max_raw_mtu: maximum MTU size for raw packets.
 * @min_raw_mtu: minimum MTU size for raw packets.
 * @core_dev_minor: minor id of the core device. Only used when working with simulator.
 * @pad_size: the pad size in bytes for the skb to transmit.
 * @serialize_ports_open: open the ports one by one and in the order they were called.
 * @in_reset: is the entire NIC currently under reset.
 * @is_threaded_tx: Schedule RAW TX on thread. Used for simulator because ndo_start_xmit is a
 *                  softirq and we cannot access the HW atomically.
 * @poll_enable: Enable Rx polling rather than IRQ + NAPI.
 * @in_teardown: true if the NIC is in teardown (during device remove).
 * @is_initialized: was the device initialized successfully.
 * @sleep_after_port_close: true if need to sleep between port close and immediate port open.
 */
struct hl_en_device {
	struct pci_dev		*pdev;
	struct device		*dev;
	struct hl_en_port	*ports;
	struct hl_aux_dev	*aux_dev;
	void			*asic_specific;
	char			*driver_ver;
	char			*fw_ver;
	char			*qsfp_eeprom;
	char			*mac_addr;
	struct hl_en_asic_funcs	asic_funcs;
	enum hl_nic_asic_type	asic_type;
	u64			sb_base_addr;
	u64			sb_base_size;
	u64			swq_base_addr;
	u64			swq_base_size;
	u64			ports_mask;
	u64			auto_neg_mask;
	u32			max_num_of_ports;
	u32			port_reset_timeout;
	u32			pending_reset_long_timeout;
	u32			max_frm_len;
	u32			raw_elem_size;
	u16			max_raw_mtu;
	u16			min_raw_mtu;
	u16			core_dev_minor;
	u16			pad_size;
	u8			serialize_ports_open;
	u8			in_reset;
	u8			is_threaded_tx;
	u8			poll_enable;
	u8			in_teardown;
	u8			is_initialized;
	u8			sleep_after_port_close;
};

#ifndef _HAS_AUX_BUS_H
int hl_en_probe(struct hl_aux_dev *aux_dev);
void hl_en_remove(struct hl_aux_dev *aux_dev);
#endif

int hl_en_dev_init(struct hl_en_device *hdev);
void hl_en_dev_fini(struct hl_en_device *hdev);

extern const struct ethtool_ops hl_en_ethtool_ops;
extern const struct dcbnl_rtnl_ops hl_en_dcbnl_ops;

bool hl_en_rx_poll_start(struct hl_en_port *port);
void hl_en_rx_poll_stop(struct hl_en_port *port);
void hl_en_rx_poll_trigger_init(struct hl_en_port *port);
int hl_en_port_reset(struct hl_en_port *port);
int hl_en_port_reset_locked(struct hl_aux_dev *aux_dev, u32 port_idx);
int hl_en_handle_rx(struct hl_en_port *port, int budget);

void gaudi_en_set_aux_ops(struct hl_en_device *hdev);
void gaudi2_en_set_aux_ops(struct hl_en_device *hdev);

#endif /* HABANALABS_EN_H_ */
