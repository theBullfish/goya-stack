// SPDX-License-Identifier: GPL-2.0

/* Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 */

#include "gaudi2_en.h"
#include <linux/etherdevice.h>
#include <linux/if_vlan.h>
#include <linux/pci.h>
#include <linux/circ_buf.h>

#define TX_DONE_CNT_THRESHOLD		4

#define RING_SIZE_MASK(ring)		((ring)->count - 1)

/* MAC configuration */
#define MAC_CFG_MAC(addr, data)		mac_write(gaudi2_nic, i, "mac", addr, data)
#define MAC_CFG_MAC_CORE(addr, data)	mac_write(gaudi2_nic, i, "mac_core", addr, data)
#define MAC_CFG_XPCS(addr, data)	mac_write(gaudi2_nic, i, "xpcs", addr, data)
#define MAC_CFG_XPCS91(addr, data)	mac_write(gaudi2_nic, i, "xpcs91", addr, data)

static void req_qpc_init(struct gaudi2_qpc_requester *req_qpc, unsigned int mtu, int last_idx,
			u32 schedq_num, bool enable)
{
	REQ_QPC_SET_TRANSPORT_SERVICE(*req_qpc, TS_RAW);
	REQ_QPC_SET_LAST_IDX(*req_qpc, last_idx);
	REQ_QPC_SET_WQ_TYPE(*req_qpc, 1);
	REQ_QPC_SET_WQ_BASE_ADDR(*req_qpc, 0);
	REQ_QPC_SET_MTU(*req_qpc, mtu);
	REQ_QPC_SET_REMOTE_WQ_LOG_SZ(*req_qpc, 1);
	/* TODO: only partial write for invalidate: SW-82003 */
	REQ_QPC_SET_VALID(*req_qpc, (u64)enable);
	REQ_QPC_SET_TRUST_LEVEL(*req_qpc, SECURED);
	REQ_QPC_SET_PORT(*req_qpc, 0);
	REQ_QPC_SET_DATA_MMU_BYPASS(*req_qpc, 1);
	REQ_QPC_SET_BURST_SIZE(*req_qpc, 1);
	REQ_QPC_SET_SCHD_Q_NUM(*req_qpc, schedq_num);
}

static void res_qpc_init(struct gaudi2_qpc_responder *res_qpc, u32 raw_qpn,
			 u32 schedq_num, bool enable)
{
	RES_QPC_SET_TRANSPORT_SERVICE(*res_qpc, TS_RAW);
	/* TODO: only partial write for invalidate: SW-82003 */
	RES_QPC_SET_VALID(*res_qpc, (u64)enable);
	RES_QPC_SET_TRUST_LEVEL(*res_qpc, SECURED);
	RES_QPC_SET_PORT(*res_qpc, 0);
	RES_QPC_SET_CQ_NUM(*res_qpc, raw_qpn);
	RES_QPC_SET_DATA_MMU_BYPASS(*res_qpc, 1);
	RES_QPC_SET_SCHD_Q_NUM(*res_qpc, schedq_num);
}

static int gaudi2_en_read_pkt_from_hw(struct hl_en_port *port, void **pkt_addr, u32 *pkt_size)
{
	struct gaudi2_port *gaudi2_nic = port->asic_specific;
	struct hl_en_device *hdev = port->hdev;
	struct hl_nic_ring *rx_ring, *cq_ring;
	struct gaudi2_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	struct hl_aux_dev *aux_dev;
	u32 port_idx = port->idx;
	struct gaudi2_cqe *cqe_p;
	u32 pi, size, wqe_idx;

	gaudi2 = hdev->asic_specific;
	aux_ops = gaudi2->aux_ops;
	aux_dev = hdev->aux_dev;

	rx_ring = gaudi2_nic->rx_ring;
	cq_ring = gaudi2_nic->cq_ring;

	/* check if packet is available by reading the PI */
	if (cq_ring->ci_shadow == cq_ring->pi_shadow) {
		pi = *((u32 *)RING_PI_ADDRESS(cq_ring));
		if (pi == cq_ring->pi_shadow)
			return ETH_PKT_NONE;

		cq_ring->pi_shadow = pi;
	}

	cqe_p = (struct gaudi2_cqe *)RING_BUF_ADDRESS(cq_ring) +
		(cq_ring->ci_shadow & RING_SIZE_MASK(cq_ring));

	if (!CQE_IS_VALID(cqe_p)) {
		dev_warn_ratelimited(hdev->dev, "Port-%d got invalid CQE on CQ\n", port_idx);
		return ETH_PKT_DROP;
	}

	/* wqe index will point to the buffer consumed by HW
	 * TODO: create proper set/get inline funcs for hw fields
	 */
	wqe_idx = cqe_p->data[1] & RING_SIZE_MASK(rx_ring);
	size = cqe_p->data[3];

	if (size > hdev->max_frm_len || size == 0) {
		dev_err_ratelimited(hdev->dev, "invalid packet size %u\n", size);
		return ETH_PKT_DROP;
	}

	*pkt_addr = RING_BUF_ADDRESS(rx_ring) + wqe_idx * hdev->raw_elem_size;
	*pkt_size = size;

	cq_ring->ci_shadow++;

	/* Mark the CQ-entry is not valid */
	CQE_SET_INVALID(cqe_p);

	/* inform the HW with our current CI */
	aux_ops->write_rx_ci(aux_dev, port_idx, cq_ring->ci_shadow);

	return ETH_PKT_OK;
}

static void gaudi2_en_configure_cq(struct hl_en_port *port, bool enable)
{
	struct hl_en_device *hdev = port->hdev;
	struct gaudi2_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	struct hl_aux_dev *aux_dev;
	u32 port_idx = port->idx;

	gaudi2 = hdev->asic_specific;
	aux_ops = gaudi2->aux_ops;
	aux_dev = hdev->aux_dev;


	aux_ops->configure_cq(aux_dev, port_idx, enable);
}

static void gaudi2_en_arm_cq(struct hl_en_port *port)
{
	struct gaudi2_port *gaudi2_port = port->asic_specific;
	struct hl_en_device *hdev = port->hdev;
	struct gaudi2_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	struct hl_aux_dev *aux_dev;
	u32 port_idx = port->idx;

	gaudi2 = hdev->asic_specific;
	aux_ops = gaudi2->aux_ops;
	aux_dev = hdev->aux_dev;

	aux_ops->arm_cq(aux_dev, port_idx, gaudi2_port->cq_ring->ci_shadow + 1);
}

static int gaudi2_en_config_qp(struct hl_en_port *port, bool enable)
{
	struct gaudi2_port *gaudi2_nic = port->asic_specific;
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct gaudi2_qpc_requester req_qpc = {};
	struct gaudi2_qpc_responder res_qpc = {};
	struct gaudi2_en_core_info *core_info;
	struct hl_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	struct hl_aux_dev *aux_dev;
	u32 port_idx, raw_qpn, schedq_num;
	struct qpc_mask mask = {};
	int last_idx, rc;
	unsigned int mtu;

	gaudi2 = hdev->asic_specific;
	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	core_info = gaudi2->core_info;
	port_idx = gaudi2_nic->idx;
	raw_qpn = core_info->raw_qpn;
	schedq_num = core_info->schedq_num;
	mtu = port->ndev->mtu + HL_EN_MAX_HEADERS_SZ;

	/* Need to configure the log of the MTU value minus 1KB as this is the minimum valid MTU.
	 * If the MTU value is not a power of 2, use the next possible value.
	 */
	mtu = __fls(mtu) - 10 + !is_power_of_2(mtu);

	last_idx = gaudi2_nic->wq_ring->count - 1;

	if (!enable) {
		rc = aux_ops->eq_dispatcher_unregister_qp(aux_dev, port_idx, raw_qpn);
		if (rc) {
			netdev_err(port->ndev, "Failed to unregister QP, %d\n", rc);
			return rc;
		}

		REQ_QPC_SET_VALID(mask, 1);
		rc = aux_ops->qpc_write(aux_dev, port_idx, &req_qpc, &mask, raw_qpn, true);
		if (rc) {
			netdev_err(port->ndev, "Failed to configure requester QP, %d\n", rc);
			return rc;
		}

		memset(&mask, 0, sizeof(mask));
		RES_QPC_SET_VALID(mask, 1);
		rc = aux_ops->qpc_write(aux_dev, port_idx, &res_qpc, &mask, raw_qpn, false);
		if (rc)
			netdev_err(port->ndev, "Failed to configure responder QP, %d\n", rc);

		return rc;
	}

	memset(&res_qpc, 0, sizeof(res_qpc));
	res_qpc_init(&res_qpc, raw_qpn, schedq_num, enable);
	rc = aux_ops->qpc_write(aux_dev, port_idx, &res_qpc, NULL, raw_qpn, false);
	if (rc) {
		netdev_err(port->ndev, "Failed to configure responder QP, %d\n", rc);
		goto qpc_res_fail;
	}

	memset(&req_qpc, 0, sizeof(req_qpc));
	req_qpc_init(&req_qpc, mtu, last_idx, schedq_num, enable);
	rc = aux_ops->qpc_write(aux_dev, port_idx, &req_qpc, NULL, raw_qpn, true);
	if (rc) {
		netdev_err(port->ndev, "Failed to configure requester QP, %d\n", rc);
		return rc;
	}

	rc = aux_ops->eq_dispatcher_register_qp(aux_dev, port_idx, core_info->kernel_asid,
			raw_qpn);
	if (rc) {
		netdev_err(port->ndev, "Failed to register QP, %d\n", rc);
		goto qp_register_fail;
	}

	return 0;

qp_register_fail:
	/* here enable is true */
	RES_QPC_SET_VALID(res_qpc, 0);
	aux_ops->qpc_write(aux_dev, port_idx, &res_qpc, NULL, raw_qpn, false);
qpc_res_fail:
	memset(&req_qpc, 0, sizeof(req_qpc));
	REQ_QPC_SET_VALID(req_qpc, 0);
	aux_ops->qpc_write(aux_dev, port_idx, &req_qpc, NULL, raw_qpn, true);

	return rc;
}

static void gaudi2_en_tx_done(struct gaudi2_port *gaudi2_nic, struct hl_nic_eqe *eqe_p)
{
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct gaudi2_en_core_info *asic_core_info;
	struct hl_en_core_info *core_info;
	struct gaudi2_en_aux_ops *asic_aux_ops;
	struct gaudi2_en_tx_buf *tx_buf;
	u32 port_idx = gaudi2_nic->idx, raw_qpn, handled_ci, pi;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	struct net_device *ndev;

	ndev = hdev->ports[port_idx].ndev;
	aux_dev = hdev->aux_dev;
	core_info = aux_dev->core_info;
	asic_core_info = core_info->asic_specific;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;
	raw_qpn = asic_core_info->raw_qpn;

	if (EQE_RAW_TX_EVENT_QPN(eqe_p) != raw_qpn) {
		netdev_warn(ndev, "tx-done: port %d got wrong QP (%d vs %d); ignoring",
			    gaudi2_nic->idx, EQE_RAW_TX_EVENT_QPN(eqe_p), raw_qpn);
		return;
	}

	if (EQE_RAW_TX_EVENT_IDX(eqe_p) >= asic_core_info->tx_ring_len) {
		netdev_err(ndev, "tx-done: port %d got invalid WQE index (%d max %d); ignoring",
			   gaudi2_nic->idx, EQE_RAW_TX_EVENT_IDX(eqe_p),
			   asic_core_info->tx_ring_len - 1);
		return;
	}

	/* Check if the index we got is in the current data bounds (indicated by the CI and PI) */
	pi = gaudi2_nic->tx_buf_info_pi;
	if (CIRC_CNT(pi, EQE_RAW_TX_EVENT_IDX(eqe_p), asic_core_info->tx_ring_len) >
			CIRC_CNT(pi, gaudi2_nic->tx_buf_info_ci, asic_core_info->tx_ring_len)) {
		dev_warn_ratelimited(hdev->dev,
				     "tx-done: port %d got stale WQE index (expecting values between %d to %d, got %d); ignoring",
				     gaudi2_nic->idx, gaudi2_nic->tx_buf_info_ci,
				     gaudi2_nic->tx_buf_info_pi, EQE_RAW_TX_EVENT_IDX(eqe_p));
		return;
	}

	if (EQE_RAW_TX_EVENT_IDX(eqe_p) != gaudi2_nic->tx_buf_info_ci)
		dev_notice_ratelimited(hdev->dev,
				       "tx-done: port-%d got miss-match in CIs (%d vs %d) on QP-%d; recovering",
				       port_idx, gaudi2_nic->tx_buf_info_ci,
				       EQE_RAW_TX_EVENT_IDX(eqe_p), EQE_RAW_TX_EVENT_QPN(eqe_p));


	/* Handle all entries up to the entry reported in the event */
	do {
		tx_buf = gaudi2_nic->tx_buf_info + gaudi2_nic->tx_buf_info_ci;
		asic_aux_ops->dma_unmap_tx_pkt(aux_dev, tx_buf->dma_addr, tx_buf->len);
		dev_kfree_skb_any(tx_buf->skb);

		handled_ci = gaudi2_nic->tx_buf_info_ci;
		gaudi2_nic->tx_buf_info_ci =
			(gaudi2_nic->tx_buf_info_ci + 1) & (asic_core_info->tx_ring_len - 1);
	} while (EQE_RAW_TX_EVENT_IDX(eqe_p) != handled_ci);
}

static uint32_t gaudi2_en_get_overrun_cnt(struct hl_aux_dev *aux_dev, u32 port_idx)
{
	struct hl_en_port *port = HL_EN_PORT(aux_dev, port_idx);
	struct gaudi2_port *gaudi2_nic;

	gaudi2_nic = port->asic_specific;

	return gaudi2_nic->fifo_overrun_err_cnt;
}

static void gaudi2_en_handle_eqe(struct hl_aux_dev *aux_dev, u32 port_idx, struct hl_nic_eqe *eqe)
{
	struct hl_en_port *port = HL_EN_PORT(aux_dev, port_idx);
	struct net_device_stats *nstats = &port->ndev->stats;
	u32 event_type = EQE_TYPE(eqe), qp, synd;
	struct hl_en_device *hdev = port->hdev;
	struct gaudi2_en_aux_ops *asic_aux_ops;
	struct gaudi2_port *gaudi2_nic;
	struct hl_en_aux_ops *aux_ops;

	gaudi2_nic = port->asic_specific;
	aux_ops = hdev->aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	switch (event_type) {
	case EQE_COMP:
#if HL_EN_DEBUG
		dev_dbg_ratelimited(hdev->dev, "Port-%d comp event\n", port_idx);
#endif
	if (!hdev->poll_enable) {
		/* napi_schedule() eventually calls __raise_softirq_irqoff() which sets the net Rx
		 * softirq to run. Since we are in thread context here, the pending softirq flag
		 * won't be checked and the Rx softirq won't be invoked. Hence we need to use the
		 * bh_disable/enable pair to invoke it.
		 */
		local_bh_disable();
		napi_schedule(&port->napi);
		local_bh_enable();
	} else {
		hl_en_rx_poll_start(port);
	}
	break;
	case EQE_RAW_TX_COMP:
#if HL_EN_DEBUG
		dev_dbg_ratelimited(hdev->dev, "Port-%d tx-comp event\n", port_idx);
#endif
		gaudi2_en_tx_done(gaudi2_nic, eqe);
		break;
	case EQE_COMP_ERR:
		net_warn_ratelimited("Port-%d cq-err event CQ:%d PI:0x%x\n",
				     port_idx, EQE_CQ_EVENT_CQ_NUM(eqe), EQE_CQ_EVENT_PI(eqe));

		nstats->rx_dropped++;
		break;
	case EQE_QP_ERR:
		synd = EQE_QP_EVENT_ERR_SYND(eqe);
		qp = EQE_QP_EVENT_QPN(eqe);
		dev_warn_ratelimited(hdev->dev, "Port-%d qp-err event QP:%d err:%d %s\n",
				     port_idx, qp, synd, asic_aux_ops->qp_err_syndrom_to_str(synd));

		/* In case of QP error we need to reset the port. We are calling the "locked"
		 * version of that function since the port->control_lock has been already
		 * taken at the beginning of the EQ handler.
		 */
		hl_en_port_reset_locked(aux_dev, port_idx);
		break;
	case EQE_DB_FIFO_OVERRUN:
		dev_warn_ratelimited(hdev->dev, "Port-%d db-fifo overrun event\n", port_idx);
		gaudi2_nic->fifo_overrun_err_cnt++;
		nstats->tx_dropped++;
		break;
	default:
		dev_warn_ratelimited(hdev->dev, "Port-%d unsupported event type: %d", port_idx,
				     event_type);
		break;
	}
}

static netdev_tx_t gaudi2_en_write_pkt_to_hw(struct hl_en_port *port, struct sk_buff *skb)
{
	u32 port_idx, tx_buf_info_pi, tx_buf_info_ci, pi, space_left_in_qp;
	struct gaudi2_port *gaudi2_nic = port->asic_specific;
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct gaudi2_en_core_info *asic_core_info;
	struct hl_en_core_info *core_info;
	struct gaudi2_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	struct hl_nic_ring *wq_ring;
	struct gaudi2_sq_wqe *wqe_p;
	struct hl_aux_dev *aux_dev;
	dma_addr_t dma_addr;
	int rc;

	tx_buf_info_pi = gaudi2_nic->tx_buf_info_pi;
	tx_buf_info_ci = gaudi2_nic->tx_buf_info_ci;
	port_idx = port->idx;
	gaudi2 = hdev->asic_specific;
	aux_ops = gaudi2->aux_ops;
	aux_dev = hdev->aux_dev;
	core_info = aux_dev->core_info;
	asic_core_info = core_info->asic_specific;

#if HL_EN_DEBUG
		dev_dbg_ratelimited(hdev->dev,
				    "port: %d, addr: 0x%p, size: %d, tx_pi: %d, tx_ci: %d\n",
				    port_idx, skb->data, skb->len, tx_buf_info_pi, tx_buf_info_ci);
#endif

	wq_ring = gaudi2_nic->wq_ring;

	/* TODO: SW-82278 Utilize fifo space calculation to use all entries */
	/* Check here if we have enough space on the QP WQ */
	space_left_in_qp = ((tx_buf_info_pi >= tx_buf_info_ci) ?
				(asic_core_info->tx_ring_len - (tx_buf_info_pi - tx_buf_info_ci)) :
						(tx_buf_info_ci - tx_buf_info_pi)) - 1;
	if (!space_left_in_qp) {
		dev_err_ratelimited(hdev->dev, "port %d QP full. PI %d, CI %d\n",
				    port_idx, tx_buf_info_pi, tx_buf_info_ci);
		if (hdev->is_threaded_tx)
			goto threaded_tx_err_exit;

		return NETDEV_TX_BUSY;
	}

	dma_addr = aux_ops->dma_map_tx_pkt(aux_dev, skb->data, skb->len);

	if (unlikely(dma_mapping_error(hdev->dev, dma_addr))) {
		dev_err_ratelimited(hdev->dev, "port %d failed to map DMA address\n", port_idx);

		dev_kfree_skb_any(skb);
		return NETDEV_TX_OK;
	}

	gaudi2_nic->tx_buf_info[tx_buf_info_pi].dma_addr = dma_addr;
	gaudi2_nic->tx_buf_info[tx_buf_info_pi].skb = skb;
	gaudi2_nic->tx_buf_info[tx_buf_info_pi].len = skb->len;
	gaudi2_nic->tx_buf_info_pi = (tx_buf_info_pi + 1) & (asic_core_info->tx_ring_len - 1);

	/* point on the next WQE */
	pi = wq_ring->pi_shadow;
	wq_ring->pi_shadow = (wq_ring->pi_shadow + 1) & (wq_ring->count - 1);

	wqe_p = (struct gaudi2_sq_wqe *)RING_BUF_ADDRESS(wq_ring) + pi;
	memset(wqe_p, 0, sizeof(*wqe_p));

	/* for ethernet only, turn on the solicite event bit */
	CFG_SQ_WQE_RESET(wqe_p);
	CFG_SQ_WQE_OPCODE(wqe_p, WQE_LINEAR);
	CFG_SQ_WQE_INDEX(wqe_p, pi & 0xff);
	CFG_SQ_WQE_INLINE(wqe_p, 0);
	CFG_SQ_WQE_LOCAL_ADDRESS(wqe_p, dma_addr);
	CFG_SQ_WQE_SIZE(wqe_p, (u64)skb->len);

	 /* TODO: SW-25778 Implement interrupt coalescing for raw pkts */
	CFG_SQ_WQE_SOL_EVENT(wqe_p, 1);

	/* make sure data is filled before ringing the db */
	mb();

	/* Ring doorbell */
	rc = aux_ops->ring_tx_doorbell(aux_dev, port_idx, wq_ring->pi_shadow);
	if (rc) {
		/* revert indices, unmap the skb and return the error */
		gaudi2_nic->tx_buf_info_pi = tx_buf_info_pi;
		wq_ring->pi_shadow = pi;
		aux_ops->dma_unmap_tx_pkt(aux_dev, dma_addr, skb->len);
		gaudi2_nic->tx_buf_info[tx_buf_info_pi].skb = NULL;
		if (hdev->is_threaded_tx)
			goto threaded_tx_err_exit;

		return NETDEV_TX_BUSY;
	}

	return NETDEV_TX_OK;

threaded_tx_err_exit:
	/* in threaded TX we (eventually) return success which will cause the SKB not to be
	 * freed so we need to free it here.
	 */
	dev_kfree_skb_any(skb);
	return NETDEV_TX_OK;
}

static int gaudi2_en_port_open(struct hl_en_port *port)
{
	struct gaudi2_port *gaudi2_nic = port->asic_specific;
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct hl_nic_ring *wq_ring, *cq_ring;
	struct gaudi2_device *gaudi2;
	int rc;

	gaudi2 = hdev->asic_specific;

	/* Reset Tx ring shadow PI/CI */
	wq_ring = gaudi2_nic->wq_ring;
	wq_ring->pi_shadow = 0;
	wq_ring->ci_shadow = 0; /* Unused */

	/* Reset SW Tx buffer PI/CI */
	gaudi2_nic->tx_buf_info_pi = 0;
	gaudi2_nic->tx_buf_info_ci = 0;

	/* Reset FIFO overrun error counter */
	gaudi2_nic->fifo_overrun_err_cnt = 0;

	/* Reset CQ ring HW PI and shadow PI/CI */
	cq_ring = gaudi2_nic->cq_ring;
	*((u32 *)RING_PI_ADDRESS(cq_ring)) = 0;
	cq_ring->pi_shadow = 0;
	cq_ring->ci_shadow = 0;

	rc = gaudi2_en_config_qp(port, true);
	if (rc) {
		netdev_warn(port->ndev, "Failed to configure QPs, %d\n", rc);
		return rc;
	}

	/* We would need the CQ-ARM for both polling and NAPI flows. This is because, even in
	 * polling mode, we would start the Rx Poll only upon the CQ-ARM event triggering the EQ
	 * for Rx completion.
	 */
	gaudi2_en_configure_cq(port, true);
	gaudi2_en_arm_cq(port);

	return 0;
}

static void gaudi2_en_db_fifo_reset(struct gaudi2_port *gaudi2_nic)
{
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct gaudi2_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;

	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	asic_aux_ops->db_fifo_reset(aux_dev, gaudi2_nic->idx);
}

static void gaudi2_en_flush_tx_buffer(struct gaudi2_port *gaudi2_nic)
{
	struct hl_en_device *hdev = gaudi2_nic->hdev;
	struct gaudi2_en_core_info *asic_core_info;
	struct gaudi2_en_aux_ops *asic_aux_ops;
	struct hl_en_core_info *core_info;
	struct gaudi2_en_tx_buf *tx_buf;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	u32 ci, pi;

	aux_dev = hdev->aux_dev;
	core_info = aux_dev->core_info;
	asic_core_info = core_info->asic_specific;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;
	ci = gaudi2_nic->tx_buf_info_ci;
	pi = gaudi2_nic->tx_buf_info_pi;

	while (ci != pi) {
		tx_buf = &gaudi2_nic->tx_buf_info[ci];
		asic_aux_ops->dma_unmap_tx_pkt(aux_dev, tx_buf->dma_addr, tx_buf->len);
		dev_kfree_skb_any(tx_buf->skb);

		ci = (ci + 1) & (asic_core_info->tx_ring_len - 1);
	}

	gaudi2_nic->tx_buf_info_ci = ci;
}

static void gaudi2_en_port_close(struct hl_en_port *port)
{
	struct gaudi2_port *gaudi2_nic = port->asic_specific;
	int rc;

	gaudi2_en_configure_cq(port, false);

	/* disable ETH Rx/Tx in H/W */
	rc = gaudi2_en_config_qp(port, false);
	if (rc)
		netdev_warn(port->ndev, "Failed to destroy QPs, %d\n", rc);

	gaudi2_en_db_fifo_reset(gaudi2_nic);

	/* Discard skbs safely from tx_buf as we won't get the tx_done call from the EQ now that the
	 * port is closed.
	 */
	gaudi2_en_flush_tx_buffer(gaudi2_nic);
}

static int gaudi2_en_dev_init(struct hl_en_device *hdev)
{
	struct hl_aux_dev *aux_dev = hdev->aux_dev;
	struct gaudi2_en_core_info *asic_core_info;
	struct gaudi2_port *gaudi2_port, *ports;
	struct gaudi2_en_aux_ops *asic_aux_ops;
	struct hl_en_core_info *core_info;
	struct hl_en_aux_ops *aux_ops;
	struct gaudi2_device *gaudi2;
	int i, rc = 0, ports_cnt = 0;
	struct hl_en_port *port;
	u32 tx_ring_size;

	core_info = aux_dev->core_info;
	asic_core_info = core_info->asic_specific;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	gaudi2 = kzalloc(sizeof(*gaudi2), GFP_KERNEL);
	if (!gaudi2)
		return -ENOMEM;

	ports = kcalloc(hdev->max_num_of_ports, sizeof(*ports), GFP_KERNEL);
	if (!ports) {
		rc = -ENOMEM;
		goto ports_alloc_fail;
	}

	tx_ring_size = asic_core_info->tx_ring_len * sizeof(struct gaudi2_en_tx_buf);

	for (i = 0 ; i < hdev->max_num_of_ports ; i++, ports_cnt++) {
		if (!(hdev->ports_mask & BIT(i)))
			continue;

		gaudi2_port = &ports[i];
		gaudi2_port->tx_buf_info = kzalloc(tx_ring_size, GFP_KERNEL);
		if (!gaudi2_port->tx_buf_info) {
			rc = -ENOMEM;
			goto ports_init_fail;
		}

		gaudi2_port->idx = i;
		gaudi2_port->hdev = hdev;
		gaudi2_port->rx_ring = asic_core_info->rx_rings[i];
		gaudi2_port->cq_ring = asic_core_info->cq_rings[i];
		gaudi2_port->wq_ring = asic_core_info->wq_rings[i];
		port = &hdev->ports[i];
		port->asic_specific = gaudi2_port;
	}

	asic_aux_ops->handle_eqe = gaudi2_en_handle_eqe;
	asic_aux_ops->port_reset_locked = hl_en_port_reset_locked;
	asic_aux_ops->get_overrun_cnt = gaudi2_en_get_overrun_cnt;

	gaudi2->ports = ports;
	gaudi2->core_info = asic_core_info;
	gaudi2->aux_ops = asic_aux_ops;

	hdev->asic_specific = gaudi2;

	/* Due to a H/W bug that even ports' link-up event arrives to the odd ports' EQ, we can't
	 * avoid obsolete events to arrive once the port is up again by flushing the H/W EQ.
	 * Instead we sleep to allows the obsolete events to be dropped while the port is closed.
	 * In hard reset flow we flush the EQ so the below doesn't needed.
	 */
	hdev->sleep_after_port_close = true;

	hdev->pad_size = gaudi2->core_info->pad_size;

	return 0;

ports_init_fail:
	for (i = 0 ; i < ports_cnt ; i++) {
		if (!(hdev->ports_mask & BIT(i)))
			continue;

		gaudi2_port = &ports[i];
		kfree(gaudi2_port->tx_buf_info);
	}
ports_alloc_fail:
	kfree(gaudi2);

	return rc;
}

static void gaudi2_en_dev_fini(struct hl_en_device *hdev)
{
	struct gaudi2_device *gaudi2 = hdev->asic_specific;

	if (!gaudi2)
		return;

	kfree(gaudi2->ports);
	kfree(gaudi2);
}

void gaudi2_en_set_aux_ops(struct hl_en_device *hdev)
{
	struct hl_en_asic_funcs *asic_funcs = &hdev->asic_funcs;

	asic_funcs->dev_init = gaudi2_en_dev_init;
	asic_funcs->dev_fini = gaudi2_en_dev_fini;
	asic_funcs->eth_port_open = gaudi2_en_port_open;
	asic_funcs->eth_port_close = gaudi2_en_port_close;
	asic_funcs->reenable_rx_irq = gaudi2_en_arm_cq;
	asic_funcs->write_pkt_to_hw = gaudi2_en_write_pkt_to_hw;
	asic_funcs->read_pkt_from_hw = gaudi2_en_read_pkt_from_hw;
	asic_funcs->get_pfc_cnts = gaudi2_en_dcbnl_get_pfc_cnts;
}
