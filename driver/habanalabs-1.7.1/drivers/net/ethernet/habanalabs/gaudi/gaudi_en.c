// SPDX-License-Identifier: GPL-2.0

/* Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 */

#include "gaudi_en.h"
#include <uapi/linux/if_ether.h>
#include <linux/etherdevice.h>
#include <linux/ipv6.h>
#include <linux/pci.h>

#if !defined(ETH_P_LLDP)
#define ETH_P_LLDP		0x88CC
#endif
#define ARP_PKT_SIZE		28
#define RAW_QPN			0

static void req_qpc_init(struct gaudi_qpc_requester *req_qp, struct gaudi_en_core_info *core_info,
			 bool enable)
{
	REQ_QPC_SET_TRANSPORT_SERVICE(*req_qp, TS_RAW);
	REQ_QPC_SET_LAST_IDX(*req_qp, (core_info->tx_ring_len - 1));
	REQ_QPC_SET_WQ_BASE_ADDR(*req_qp, core_info->tx_wq_base_addr);
	/* TODO: only partial write for invalidate: SW-82003 */
	REQ_QPC_SET_VALID(*req_qp, (u64)enable);
	REQ_QPC_SET_SECURED(*req_qp, SECURED);
	REQ_QPC_SET_PORT(*req_qp, 0);
}

static void res_qpc_init(struct gaudi_qpc_responder *res_qp, struct gaudi_en_core_info *core_info,
			 bool enable)
{
	RES_QPC_SET_TRANSPORT_SERVICE(*res_qp, TS_RAW);
	RES_QPC_SET_LOG_BUF_SIZE_MASK(*res_qp, ilog2(core_info->rx_ring_len));
	/* TODO: only partial write for invalidate: SW-82003 */
	RES_QPC_SET_VALID(*res_qp, (u64)enable);
	RES_QPC_SET_SECURED(*res_qp, SECURED);
	RES_QPC_SET_PORT(*res_qp, 0);
}

static netdev_tx_t gaudi_en_write_pkt_to_hw(struct hl_en_port *port, struct sk_buff *skb)
{
	u64 swq_addr, sb_base_address, swq_base_addr, *swq_p, *data = (u64 *)skb->data,
		hbm_bar_cur_addr;
	struct gaudi_port *gaudi_nic = port->asic_specific;
	struct hl_en_device *hdev = port->hdev;
	u32 port_idx, pi, ci, diff, new_pi;
	struct gaudi_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	struct gaudi_device *gaudi;
	struct gaudi_sq_wqe swq;
	int i;

	aux_dev = hdev->aux_dev;
	gaudi = hdev->asic_specific;
	aux_ops = gaudi->aux_ops;
	pi = gaudi_nic->tx_pi;
	ci = gaudi_nic->tx_ci;
	port_idx = port->idx;

#if HL_EN_DEBUG
		dev_dbg_ratelimited(hdev->dev,
				    "port: %d, addr: 0x%p, size: %d, tx_pi: %d, tx_ci: %d\n",
				    port_idx, skb->data, skb->len, pi, ci);
#endif

	swq_p = (u64 *)&swq;

	if (pi >= ci)
		diff = pi - ci;
	else
		diff = gaudi->tx_queue_len - ci + pi;

	/* update CI once in a while */
	if (diff > (gaudi->tx_queue_len >> 1)) {
		gaudi_nic->tx_ci = aux_ops->read_tx_ci(aux_dev, port_idx);
		ci = gaudi_nic->tx_ci;
	}

	new_pi = (pi + 1) & (gaudi->tx_queue_len - 1);
	if (new_pi == ci)
		return NETDEV_TX_BUSY;

	gaudi_nic->tx_pi = new_pi;

	sb_base_address = hdev->sb_base_addr + port_idx * hdev->sb_base_size +
								pi * hdev->raw_elem_size;
	swq_base_addr = hdev->swq_base_addr + port_idx * hdev->swq_base_size;

	/* Create SWQ */
	memset(&swq, 0, sizeof(swq));
	CFG_SQ_WQE_OPCODE(swq, WQE_LINEAR);
	CFG_SQ_WQE_LOCAL_ADDRESS_31_0(swq, sb_base_address & 0xFFFFFFFF);
	CFG_SQ_WQE_LOCAL_ADDRESS_49_32(swq, (sb_base_address >> 32) & 0x3FFFF);
	CFG_SQ_WQE_SIZE(swq, (u64)skb->len);

	hbm_bar_cur_addr = *gaudi->hbm_bar_cur_addr;

	/* Copy packet to SB */
	for (i = 0 ; i < DIV_ROUND_UP_ULL(skb->len, sizeof(u64)) ; i++)
		writeq(data[i], gaudi->pcie_bar + ((sb_base_address + i * 8) - hbm_bar_cur_addr));

	/* Copy WQE to SWQ Buffer */
	for (i = 0 ; i < DIV_ROUND_UP_ULL(sizeof(swq), sizeof(u64)) ; i++) {
		swq_addr = swq_base_addr + (pi * sizeof(struct gaudi_sq_wqe) + i * 8);
		writeq(swq_p[i], gaudi->pcie_bar + (swq_addr - hbm_bar_cur_addr));
	}

	/* Perform read to flush the writes */
	readq(gaudi->pcie_bar + sb_base_address - hbm_bar_cur_addr);

	/* Make sure we ring the doorbell after the data copying */
	mb();

	/* Ring doorbell */
	aux_ops->ring_tx_doorbell(aux_dev, port_idx, new_pi);

	dev_kfree_skb_any(skb);

	return NETDEV_TX_OK;
}

static int gaudi_en_read_pkt_from_hw(struct hl_en_port *port, void **ppkt_addr, u32 *ppkt_size)
{
	u32 *pi, *ci, ether_type, tpid, ipv4_len, ipv6_len, payload_size, hdr_size = ETH_HLEN,
		port_idx;
	struct gaudi_port *gaudi_nic = port->asic_specific;
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	bool vlan_double_tag = false;
	struct hl_aux_dev *aux_dev;
	struct gaudi_device *gaudi;
	void *pkt_addr;
	__be32 *data;
	int idx;

	gaudi = hdev->asic_specific;
	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;
	pi = &gaudi_nic->rx_pi;
	ci = &gaudi_nic->rx_ci;
	port_idx = port->idx;

	/* check if packet is available by reading the PI, but do it only if
	 * needed as it is expensive
	 */
	if (*pi == *ci) {
		*pi = asic_aux_ops->read_rx_pi(aux_dev, port_idx);
		if (*pi == *ci)
			return ETH_PKT_NONE;
	}

	pkt_addr = gaudi_nic->rx_mem_addr + (*ci) * hdev->raw_elem_size;

	/* At this point we have a packet to handle that will be received or skipped. Either way we
	 * need to increment the CI.
	 */
	*ci = (*ci + 1) & (gaudi->rx_mem_len - 1);
	data = (__be32 *)pkt_addr;

	/* skip MAC header */
	idx = (ETH_ALEN * 2) / 4;

	/* handle QinQ */
	tpid = ntohl(data[idx++]) >> 16;
	if (tpid == ETH_P_8021AD) {
		/* skip outer VLAN tag */
		tpid = ntohl(data[idx++]) >> 16;
		vlan_double_tag = true;
		hdr_size += 4;
	}

	/* handle VLAN */
	if (tpid == ETH_P_8021Q) {
		/* skip VLAN tag */
		ether_type = ntohl(data[idx++]) >> 16;
		hdr_size += 4;
	} else if (vlan_double_tag) {
		dev_dbg_ratelimited(hdev->dev, "error, wrong inner TPID (0x%x) in QinQ\n", tpid);
		ether_type = UINT_MAX;
	} else {
		ether_type = tpid;
	}

	if (ether_type <= ETH_DATA_LEN) {
		payload_size = ether_type;
		/* In case the Ethernet type represents length, the total length
		 * must be at least 60 bytes as Ethernet frames are at least
		 * 60 bytes excluding FCS
		 */
		if ((hdr_size + payload_size) < ETH_ZLEN) {
			dev_dbg_ratelimited(hdev->dev,
					    "error, unsupported EtherType 0x%x, port %d\n",
					    ether_type, port_idx);
			return ETH_PKT_DROP;
		}
	} else if (ether_type == ETH_P_ARP) {
		payload_size = ARP_PKT_SIZE;
	} else if (ether_type == ETH_P_IP) {
		ipv4_len = ntohl(data[idx]) >> 16;
		payload_size = ipv4_len;
	} else if (ether_type == ETH_P_IPV6) {
		ipv6_len = ntohl(data[idx]) & 0xFFFF;
		payload_size = ipv6_len + sizeof(struct ipv6hdr);
	} else if ((ether_type == ETH_P_LLDP) ||
			(ether_type == ETH_P_LOOPBACK) ||
			(ether_type == ETH_P_802_EX1)) {
		payload_size = ETH_DATA_LEN;
	} else {
		dev_dbg_ratelimited(hdev->dev, "error, unsupported EtherType 0x%x, port %d\n",
				    ether_type, port_idx);
		return ETH_PKT_DROP;
	}

	if (payload_size > port->ndev->mtu) {
		dev_dbg_ratelimited(hdev->dev,
				    "error, payload size %uB exceeds MTU (%uB), EtherType 0x%x, hdr_size 0x%x, port %d\n",
				    payload_size, port->ndev->mtu, ether_type, hdr_size, port_idx);
		return ETH_PKT_DROP;
	}

#if HL_EN_DEBUG
	dev_dbg_ratelimited(hdev->dev, "port %d packet_size %d ether_type 0x%x\n",
			    port_idx, pkt_size, ether_type);
#endif

	*ppkt_addr = pkt_addr;
	*ppkt_size = hdr_size + payload_size;

	return ETH_PKT_OK;
}

static void gaudi_en_reenable_rx_irq(struct hl_en_port *port)
{
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	u32 port_idx = port->idx;

	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	asic_aux_ops->reenable_rx_irq(aux_dev, port_idx);
}

static irqreturn_t __gaudi_en_rx_irq_handler(struct hl_en_port *port)
{
	napi_schedule(&port->napi);

	return IRQ_HANDLED;
}

static irqreturn_t gaudi_en_rx_irq_handler_aux(struct hl_aux_dev *aux_dev, u32 port_idx)
{
	struct hl_en_port *port = HL_EN_PORT(aux_dev, port_idx);

	return __gaudi_en_rx_irq_handler(port);
}

static irqreturn_t gaudi_en_rx_irq_handler(int irq, void *arg)
{
	struct hl_en_port *port = arg;

	return __gaudi_en_rx_irq_handler(port);
}

static int gaudi_en_config_qp(struct hl_en_port *port, bool enable)
{
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_core_info *core_info;
	struct gaudi_qpc_requester req_qp = {};
	struct gaudi_qpc_responder res_qp = {};
	struct hl_en_aux_ops *aux_ops;
	struct gaudi_device *gaudi;
	struct hl_aux_dev *aux_dev;
	struct qpc_mask mask = {};
	u32 port_idx = port->idx;
	int rc;

	gaudi = hdev->asic_specific;
	core_info = gaudi->core_info;
	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;

	if (!enable) {
		REQ_QPC_SET_VALID(mask, 1);
		rc = aux_ops->qpc_write(aux_dev, port_idx, &req_qp.data, &mask, RAW_QPN, true);
		if (rc) {
			netdev_err(port->ndev, "Failed to write requester QP, %d\n", rc);
			return rc;
		}

		memset(&mask, 0, sizeof(mask));
		RES_QPC_SET_VALID(mask, 1);
		rc = aux_ops->qpc_write(aux_dev, port_idx, &res_qp.data, &mask, RAW_QPN, false);
		if (rc)
			netdev_err(port->ndev, "Failed to write responder QP, %d\n", rc);

		return rc;
	}

	memset(&res_qp, 0, sizeof(res_qp));
	res_qpc_init(&res_qp, core_info, enable);
	rc = aux_ops->qpc_write(aux_dev, port_idx, &res_qp.data, NULL, RAW_QPN, false);
	if (rc) {
		netdev_err(port->ndev, "Failed to write responder QP, %d\n", rc);
		goto qpc_res_fail;
	}

	memset(&req_qp, 0, sizeof(req_qp));
	req_qpc_init(&req_qp, core_info, enable);
	rc = aux_ops->qpc_write(aux_dev, port_idx, &req_qp.data, NULL, RAW_QPN, true);
	if (rc) {
		netdev_err(port->ndev, "Failed to write requester QP, %d\n", rc);
		return rc;
	}

	return 0;

qpc_res_fail:
	memset(&req_qp, 0, sizeof(req_qp));
	REQ_QPC_SET_VALID(req_qp, 0);
	aux_ops->qpc_write(aux_dev, port_idx, &req_qp.data, NULL, RAW_QPN, true);

	return rc;
}

static void gaudi_en_rx_poll_schedule(struct hl_en_port *port, unsigned long delay)
{
	queue_delayed_work(port->rx_wq, &port->rx_poll_work, delay);
}

static void gaudi_en_rx_poll_work(struct work_struct *work)
{
	struct hl_en_port *port = container_of(work, struct hl_en_port, rx_poll_work.work);

	hl_en_handle_rx(port, NAPI_POLL_WEIGHT);

	gaudi_en_rx_poll_schedule(port, msecs_to_jiffies(1));
}

static void gaudi_en_rx_poll_start(struct hl_en_port *port)
{
	gaudi_en_rx_poll_schedule(port, msecs_to_jiffies(1));
}

static int gaudi_en_port_open(struct hl_en_port *port)
{
	struct gaudi_port *gaudi_nic = port->asic_specific;
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	struct gaudi_device *gaudi;
	u32 port_idx = port->idx;
	int rc, irq;

	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;
	gaudi = hdev->asic_specific;

	memset(gaudi_nic->rx_mem_addr, 0, gaudi->rx_mem_len * hdev->raw_elem_size);

	if (!hdev->poll_enable) {
		if (hdev->pdev && gaudi->multi_msi_mode) {
			irq = pci_irq_vector(hdev->pdev, gaudi->base_rx_irq + port_idx);
			rc = request_irq(irq, gaudi_en_rx_irq_handler, 0, port->ndev->name,
					 port);
			if (rc) {
				netdev_err(port->ndev, "Failed to request Rx IRQ %d, %d\n",
					   irq, rc);
				return rc;
			}
		}

		asic_aux_ops->configure_rx_irq(aux_dev, port_idx, true);
	} else {
		INIT_DELAYED_WORK(&port->rx_poll_work, gaudi_en_rx_poll_work);
		gaudi_en_rx_poll_start(port);
	}

	gaudi_nic->rx_pi = 0;
	gaudi_nic->rx_ci = 0;
	gaudi_nic->tx_pi = 0;
	gaudi_nic->tx_ci = 0;

	rc = gaudi_en_config_qp(port, true);
	if (rc) {
		netdev_err(port->ndev, "Failed to configure QPs, %d\n", rc);
		goto free_irq;
	}

	return 0;

free_irq:
	if (!hdev->poll_enable) {
		asic_aux_ops->configure_rx_irq(aux_dev, port_idx, false);

		if (hdev->pdev) {
			if (gaudi->multi_msi_mode) {
				irq = pci_irq_vector(hdev->pdev, gaudi->base_rx_irq + port_idx);
				synchronize_irq(irq);
				free_irq(irq, aux_dev);
			} else {
				irq = pci_irq_vector(hdev->pdev, 0);
				synchronize_irq(irq);
			}
		}
	} else {
		hl_en_rx_poll_stop(port);
	}

	return rc;
}

static void gaudi_en_port_close(struct hl_en_port *port)
{
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	struct gaudi_device *gaudi;
	u32 port_idx = port->idx;
	int irq, rc;

	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;
	gaudi = hdev->asic_specific;

	/* disable ETH Rx/Tx QPs */
	rc = gaudi_en_config_qp(port, false);
	if (rc)
		netdev_err(port->ndev, "Failed to destroy QPs, %d\n", rc);

	/* disable Rx H/W */
	if (!hdev->poll_enable) {
		asic_aux_ops->configure_rx_irq(aux_dev, port_idx, false);

		if (hdev->pdev) {
			if (gaudi->multi_msi_mode) {
				irq = pci_irq_vector(hdev->pdev, gaudi->base_rx_irq + port_idx);
				synchronize_irq(irq);
				free_irq(irq, aux_dev);
			} else {
				irq = pci_irq_vector(hdev->pdev, 0);
				synchronize_irq(irq);
			}
		}
	}
}

static int gaudi_en_dev_init(struct hl_en_device *hdev)
{
	struct hl_aux_dev *aux_dev = hdev->aux_dev;
	struct gaudi_en_core_info *asic_core_info;
	struct gaudi_port *gaudi_port, *ports;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_core_info *core_info;
	struct hl_en_aux_ops *aux_ops;
	struct gaudi_device *gaudi;
	struct hl_en_port *port;
	int i, rc;

	core_info = aux_dev->core_info;
	asic_core_info = core_info->asic_specific;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	gaudi = kzalloc(sizeof(*gaudi), GFP_KERNEL);
	if (!gaudi)
		return -ENOMEM;

	ports = kcalloc(hdev->max_num_of_ports, sizeof(*ports), GFP_KERNEL);
	if (!ports) {
		rc = -ENOMEM;
		goto ports_alloc_fail;
	}

	for (i = 0 ; i < hdev->max_num_of_ports ; i++) {
		if (!(hdev->ports_mask & BIT(i)))
			continue;

		gaudi_port = &ports[i];
		gaudi_port->idx = i;
		gaudi_port->hdev = hdev;
		gaudi_port->rx_mem_addr = asic_core_info->rx_mem_addr[i];
		port = &hdev->ports[i];
		port->asic_specific = gaudi_port;
	}

	if (!hdev->poll_enable)
		asic_aux_ops->rx_irq_handler = gaudi_en_rx_irq_handler_aux;

	gaudi->ports = ports;
	gaudi->pcie_bar = asic_core_info->pcie_bar;
	gaudi->hbm_bar_cur_addr = asic_core_info->hbm_bar_cur_addr;
	gaudi->tx_wq_base_addr = asic_core_info->tx_wq_base_addr;
	gaudi->tx_queue_len = asic_core_info->tx_ring_len;
	gaudi->rx_mem_len = asic_core_info->rx_ring_len;
	gaudi->base_rx_irq = asic_core_info->base_rx_irq;
	gaudi->multi_msi_mode = asic_core_info->multi_msi_mode;
	gaudi->core_info = asic_core_info;
	gaudi->aux_ops = asic_aux_ops;

	hdev->asic_specific = gaudi;

	return 0;

ports_alloc_fail:
	kfree(gaudi);

	return rc;
}

static void gaudi_en_dev_fini(struct hl_en_device *hdev)
{
	struct gaudi_device *gaudi = hdev->asic_specific;

	if (!gaudi)
		return;

	kfree(gaudi->ports);
	kfree(gaudi);
}

void gaudi_en_set_aux_ops(struct hl_en_device *hdev)
{
	struct hl_en_asic_funcs *asic_funcs = &hdev->asic_funcs;

	asic_funcs->dev_init = gaudi_en_dev_init;
	asic_funcs->dev_fini = gaudi_en_dev_fini;
	asic_funcs->reenable_rx_irq = gaudi_en_reenable_rx_irq;
	asic_funcs->eth_port_open = gaudi_en_port_open;
	asic_funcs->eth_port_close = gaudi_en_port_close;
	asic_funcs->write_pkt_to_hw = gaudi_en_write_pkt_to_hw;
	asic_funcs->read_pkt_from_hw = gaudi_en_read_pkt_from_hw;
	asic_funcs->get_pfc_cnts = gaudi_en_dcbnl_get_pfc_cnts;
}
