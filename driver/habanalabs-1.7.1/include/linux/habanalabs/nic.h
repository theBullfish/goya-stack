/* SPDX-License-Identifier: GPL-2.0
 *
 * Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef HL_NIC_H_
#define HL_NIC_H_

#include <linux/types.h>
#include <linux/sizes.h>
#include <linux/irqreturn.h>
#ifdef _HAS_AUX_BUS_H
#include <linux/auxiliary_bus.h>
#endif
#include <uapi/linux/ethtool.h>

#define HL_EN_PFC_PRIO_NUM	4
#define HL_EN_MAX_HEADERS_SZ	(ETH_HLEN + 2 * VLAN_HLEN + ETH_FCS_LEN)

struct hl_aux_dev;
struct qpc_mask;

/**
 * enum hl_nic_asic_type - supported ASIC types.
 * @ASIC_GAUDI: Gaudi device.
 * @ASIC_GAUDI2: Gaudi2 device.
 */
enum hl_nic_asic_type {
	HL_ASIC_GAUDI,
	HL_ASIC_GAUDI2
};

/**
 * enum hl_en_pflags - mutable capabilities of the port.
 * PFLAGS_PCS_LINK_CHECK: check for PCS link periodically.
 * PFLAGS_PHY_AUTO_NEG_LPBK: allow Autonegotiation in loopback.
 */
enum hl_en_pflags {
	PFLAGS_PCS_LINK_CHECK = BIT(0),
	PFLAGS_PHY_AUTO_NEG_LPBK = BIT(1),
};

enum hl_ts_type {
	TS_RC = 0,
	TS_RAW = 1
};

enum hl_trust_level {
	UNSECURED = 0,
	SECURED = 1,
	PRIVILEGE = 2
};

/**
 * struct hl_en_stat - Holds ASIC specific statistics string and default register offset.
 * @str: String name of ethtool stat.
 * @lo_offset: Register offset of the stat.
 * @hi_offset: High register offset. May be unused for some stats.
 */
struct hl_en_stat {
	char str[ETH_GSTRING_LEN];
	int lo_offset;
	int hi_offset;
};

/**
 * struct hl_nic_eqe - describes an event-queue entry
 * @data: the data each event-queue entry contains
 */
struct hl_nic_eqe {
	u32 data[4];
};

/**
 * struct hl_nic_mem_resources - memory resource used by a NIC ring.
 * @addr: virtual address of the memory.
 * @dma_addr: physical address of the memory.
 * @size: memory size.
 */
struct hl_nic_mem_resource {
	void *addr;
	dma_addr_t dma_addr;
	u32 size;
};

/**
 * struct hl_nic_ring - represents a NIC ring
 * @buf: the ring buffer memory resource.
 * @pi: the memory-resident producer index of the ring, updated by HW
 * @pi_shadow: producer shadow index - used by SW
 * @ci_shadow: consumer shadow index - used by SW
 * @rep_idx: use to count until a threshold value, like HW update
 * @asid: the asid of the ring
 * @count: the number of elements the ring can hold
 * @elem_size: the rings's element size
 */
struct hl_nic_ring {
	struct hl_nic_mem_resource buf;
	struct hl_nic_mem_resource pi;

	u32 pi_shadow;
	u32 ci_shadow;

	u32 rep_idx;
	u32 asid;
	u32 count;
	u32 elem_size;
};

/* NIC ring support */
#define RING_BUF_DMA_ADDRESS(ring)	((ring)->buf.dma_addr)
#define RING_BUF_ADDRESS(ring)		((ring)->buf.addr)
#define RING_BUF_SIZE(ring)		((ring)->buf.size)
#define RING_PI_DMA_ADDRESS(ring)	((ring)->pi.dma_addr)
#define RING_PI_ADDRESS(ring)		((ring)->pi.addr)
#define RING_PI_SIZE(ring)		((ring)->pi.size)
#define RING_CI_ADDRESS(ring)		RING_BUF_ADDRESS(ring)

/* Ethernet */

/**
 * struct hl_en_core_info - habanalabs core info for the eth device.
 * @pdev: pointer to PCI device, can be NULL in case of simulator device.
 * @dev: related kernel basic device structure.
 * @asic_specific: ASIC specific core info.
 * @driver_ver: Kernel driver version.
 * @fw_ver: FW version.
 * @qsfp_eeprom: QSFPD EEPROM info.
 * @mac_addr: array of all MAC addresses.
 * @asic_type: ASIC specific type.
 * @sb_base_addr: the base address of a Tx cyclic buffer.
 * @sb_base_size: the size of a Tx eth pkt cyclic buffer.
 * @swq_base_addr: the base address of a Tx workqueue cyclic buffer
 * @swq_base_size: the size of a Tx workqueue cyclic buffer
 * @ports_mask: mask of available ports.
 * @auto_neg_mask: mask of port with Autonegotiation enabled.
 * @max_num_of_ports: max number of available ports.
 * @pending_reset_long_timeout: long timeout for pending hard reset to finish in seconds.
 * @max_frm_len: maximum allowed frame length.
 * @raw_elem_size: size of element in raw buffers.
 * @max_raw_mtu: maximum MTU size for raw packets.
 * @min_raw_mtu: minimum MTU size for raw packets.
 * @minor: minor id of the core device.
 */
struct hl_en_core_info {
	struct pci_dev		*pdev;
	struct device		*dev;
	void			*asic_specific;
	char			*driver_ver;
	char			*fw_ver;
	char			*qsfp_eeprom;
	char			**mac_addr;
	enum hl_nic_asic_type	asic_type;
	u64			sb_base_addr;
	u64			sb_base_size;
	u64			swq_base_addr;
	u64			swq_base_size;
	u64			ports_mask;
	u64			auto_neg_mask;
	u32			max_num_of_ports;
	u32			pending_reset_long_timeout;
	u32			max_frm_len;
	u32			raw_elem_size;
	u16			max_raw_mtu;
	u16			min_raw_mtu;
	u16			minor;
};

/**
 * struct hl_en_aux_ops - pointer functions for core <-> eth drivers communication.
 * @device_operational: is device operational.
 * @hw_access_lock: prevent HW access.
 * @hw_access_unlock: allow HW access.
 * @is_eth_lpbk: is Ethernet loopback enabled.
 * @port_hw_init: port HW init.
 * @port_hw_fini: port HW cleanup.
 * @phy_init: port PHY init.
 * @phy_fini: port PHY cleanup.
 * @set_pfc: enable/disable PFC.
 * @get_cnts_num: get the number of available counters.
 * @get_cnts_names: get the names of the available counters.
 * @get_cnts_values: get the values of the available counters.
 * @eq_dispatcher_register_qp: register QP to its event dispatch queue.
 * @eq_dispatcher_unregister_qp: un-register QP from its event dispatch queue.
 * @get_speed: get the port speed in Mb/s.
 * @ports_reopen: reopen the ports after hard reset.
 * @ports_stop_prepare: prepare the ports for a stop.
 * @ports_stop: stop traffic.
 * @set_port_status: set the link port status.
 * @get_mac_lpbk: get MAC loopback status.
 * @set_mac_lpbk: set MAC loopback status.
 * @update_mtu: update all QPs to use the new MTU value.
 * @qpc_write: write a QP context to the HW.
 * @ctrl_lock: control mutex lock.
 * @ctrl_unlock: control mutex unlock.
 * @is_port_open: is port open;
 * @get_src_ip: get the source IP of the given port.
 * @reset_stats: reset port statistics (called from debugfs only).
 * @get_mtu: get the port MTU value.
 * @get_pflags: get the port private flags.
 * @set_dev_lpbk: set loopback status on the net-device.
 * @get_netdev: get the netdev structure for the port
 * @asic_ops: pointer for ASIC specific ops struct.
 */
struct hl_en_aux_ops {
	/* eth2core */
	bool (*device_operational)(struct hl_aux_dev *aux_dev);
	void (*hw_access_lock)(struct hl_aux_dev *aux_dev);
	void (*hw_access_unlock)(struct hl_aux_dev *aux_dev);
	bool (*is_eth_lpbk)(struct hl_aux_dev *aux_dev);
	int (*port_hw_init)(struct hl_aux_dev *aux_dev, u32 port);
	void (*port_hw_fini)(struct hl_aux_dev *aux_dev, u32 port);
	int (*phy_init)(struct hl_aux_dev *aux_dev, u32 port);
	void (*phy_fini)(struct hl_aux_dev *aux_dev, u32 port);
	int (*set_pfc)(struct hl_aux_dev *aux_dev, u32 port, bool enable);
	int (*get_cnts_num)(struct hl_aux_dev *aux_dev, u32 port);
	void (*get_cnts_names)(struct hl_aux_dev *aux_dev, u32 port, u8 *data);
	void (*get_cnts_values)(struct hl_aux_dev *aux_dev, u32 port, u64 *data);
	bool (*get_mac_lpbk)(struct hl_aux_dev *aux_dev, u32 port);
	int (*set_mac_lpbk)(struct hl_aux_dev *aux_dev, u32 port, bool enable);
	int (*update_mtu)(struct hl_aux_dev *aux_dev, u32 port, u32 mtu);
	int (*qpc_write)(struct hl_aux_dev *aux_dev, u32 port, void *qpc,
			 struct qpc_mask *qpc_mask, u32 qpn, bool is_req);
	void (*ctrl_lock)(struct hl_aux_dev *aux_dev, u32 port);
	void (*ctrl_unlock)(struct hl_aux_dev *aux_dev, u32 port);
	int (*eq_dispatcher_register_qp)(struct hl_aux_dev *aux_dev, u32 port, u32 asid,
						u32 qp_id);
	int (*eq_dispatcher_unregister_qp)(struct hl_aux_dev *aux_dev, u32 port, u32 qp_id);
	u32 (*get_speed)(struct hl_aux_dev *aux_dev, u32 port);

	/* core2eth */
	int (*ports_reopen)(struct hl_aux_dev *aux_dev);
	void (*ports_stop_prepare)(struct hl_aux_dev *aux_dev);
	void (*ports_stop)(struct hl_aux_dev *aux_dev);
	void (*set_port_status)(struct hl_aux_dev *aux_dev, u32 port, bool up);
	bool (*is_port_open)(struct hl_aux_dev *aux_dev, u32 port);
	int (*get_src_ip)(struct hl_aux_dev *aux_dev, u32 port, u32 *src_ip);
	void (*reset_stats)(struct hl_aux_dev *aux_dev, u32 port);
	u32 (*get_mtu)(struct hl_aux_dev *aux_dev, u32 port);
	u32 (*get_pflags)(struct hl_aux_dev *aux_dev, u32 port);
	void (*set_dev_lpbk)(struct hl_aux_dev *aux_dev, u32 port, bool enable);
	struct net_device *(*get_netdev)(struct hl_aux_dev *aux_dev, u32 port);
	void *asic_ops;
};

/**
 * enum aux_dev_type - auxiliary device type.
 * HL_AUX_DEV_ETH: Ethernet.
 * HL_AUX_DEV_IB: InfiniBand.
 */
enum hl_aux_dev_type {
	HL_AUX_DEV_ETH,
	HL_AUX_DEV_IB
};

/**
 * struct hl_aux_dev - habanalabs auxiliary device structure.
 * @adev: auxiliary device.
 * @aux_ops: pointer functions for core <-> aux drivers communication.
 * @core_info: essential core data for operating the auxiliary device.
 * @priv: auxiliary device private data.
 * @type: type of the auxiliary device.
 */
struct hl_aux_dev {
#ifdef _HAS_AUX_BUS_H
	struct auxiliary_device	adev;
#endif
	void			*aux_ops;
	void			*core_info;
	void			*priv;
	enum hl_aux_dev_type	type;
};

/* InfiniBand */

/**
 * struct hl_ib_device_attr - IB device attributes.
 * @fw_ver: firmware version.
 * @max_mr_size: max size of a memory region.
 * @page_size_cap: largest page size in MMU.
 * @vendor_id: device vendor ID.
 * @vendor_part_id: device vendor part ID.
 * @hw_ver: device chip version.
 * @max_qp: max QPs supported.
 * @max_qp_wr: max QPs per work-request supported.
 * @max_cqe: max completion-queue entries supported.
 * @max_mr: max memory regions supported.
 * @phy_port_cnt: maximum physical NIC ports supported.
 */
struct hl_ib_device_attr {
	u64	fw_ver;
	u64	max_mr_size;
	u64	page_size_cap;
	u32	vendor_id;
	u32	vendor_part_id;
	u32	hw_ver;
	s32	max_qp;
	s32	max_qp_wr;
	s32	max_cqe;
	s32	max_mr;
};

/**
 * struct hl_ib_port_attr - IB port attributes.
 * @speed: speed in Mb/s.
 * @max_msg_sz: max message size
 * @max_mtu: max mtu size
 * @open: is open and fully initialized.
 * @link_up: has PCS link.
 * @num_lanes: number of lanes per port.
 */
struct hl_ib_port_attr {
	u32	speed;
	u32	max_msg_sz;
	u32	max_mtu;
	u8	open;
	u8	link_up;
	u8	num_lanes;
};

/**
 * struct hl_ib_core_info - habanalabs core info for the IB device.
 * @pdev: pointer to PCI device, can be NULL in case of simulator device.
 * @ndev: array of pointers that store the netdev pointers of each port
 * @dev: related kernel basic device structure.
 * @fw_ver: FW version.
 * @asic_type: ASIC specific type.
 * @ports_mask: mask of available ports.
 * @dram_size: available DRAM size.
 * @max_num_of_wqes: maximum number of WQ entries.
 * @pending_reset_long_timeout: long timeout for pending hard reset to finish in seconds.
 * @id: core device ID.
 * @max_num_of_ports: maximum number of ports supported by ASIC.
 */
struct hl_ib_core_info {
	struct pci_dev		*pdev;
	struct net_device	**ndev;
	struct device		*dev;
	char			*fw_ver;
	enum hl_nic_asic_type	asic_type;
	u64			ports_mask;
	u64			dram_size;
	u32			max_num_of_wqes;
	u32			pending_reset_long_timeout;
	u16			id;
	u8			max_num_of_ports;
};

/**
 * struct hl_ib_aux_ops - pointer functions for core <-> ib drivers communication.
 * @device_operational: is device operational.
 * @hw_access_lock: prevent HW access.
 * @hw_access_unlock: allow HW access.
 * @alloc_ucontext: allocate user context.
 * @dealloc_ucontext: deallocate user context.
 * @query_port: get port attributes.
 * @core_ctrl: operate the device with proprietary opcodes.
 * @query_device: get device attributes.
 */
struct hl_ib_aux_ops {
	/* ib2core */
	bool (*device_operational)(struct hl_aux_dev *aux_dev);
	void (*hw_access_lock)(struct hl_aux_dev *aux_dev);
	void (*hw_access_unlock)(struct hl_aux_dev *aux_dev);
	int (*alloc_ucontext)(struct hl_aux_dev *aux_dev, void **ctx_priv);
	void (*dealloc_ucontext)(struct hl_aux_dev *aux_dev, void *ctx_priv);
	void (*query_port)(struct hl_aux_dev *aux_dev, u32 port, struct hl_ib_port_attr *port_attr);
	int (*core_ctrl)(struct hl_aux_dev *aux_dev, void *ctx_priv, u32 op, void *input,
				void *output);
	void (*query_device)(struct hl_aux_dev *aux_dev, struct hl_ib_device_attr *device_attr);
};

#endif /* HL_NIC_H_ */
