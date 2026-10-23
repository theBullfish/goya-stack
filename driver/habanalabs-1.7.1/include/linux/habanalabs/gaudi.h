/* SPDX-License-Identifier: GPL-2.0
 *
 * Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */
#ifndef HL_GAUDI_H_
#define HL_GAUDI_H_

#include "nic.h"
#include <linux/irqreturn.h>

enum gaudi_wqe_opcode {
	WQE_NOP = 0,
	WQE_SEND = 1,
	WQE_LINEAR = 2,
	WQE_STRIDE = 3,
	WQE_MULTI_STRIDE = 4,
	WQE_RATE_UPDATE  = 5
};

struct gaudi_qpc_requester {
	__le64	data[8];
};

struct qpc_mask {
	__le64  data[sizeof(struct gaudi_qpc_requester) >> 3];
};

#define QPC_SET(qpc, idx, shift, val, len) \
		((qpc).data[idx] |= \
			cpu_to_le64((u64) ((val) & (BIT(len) - 1)) << (shift)))

#define QPC_GET(qpc, idx, shift, len) \
		(((le64_to_cpu((qpc).data[idx])) >> (shift)) & (BIT(len) - 1))

#define REQ_QPC_SET_DST_QP(req, val)		QPC_SET(req, 0, 0, val, 24)
#define REQ_QPC_SET_PORT(req, val)		QPC_SET(req, 0, 24, val, 4)
#define REQ_QPC_SET_PRIORITY(req, val)		QPC_SET(req, 0, 28, val, 2)
#define REQ_QPC_SET_CONG_EN(req, val)		QPC_SET(req, 0, 30, val, 2)
#define REQ_QPC_SET_RKEY(req, val)		QPC_SET(req, 0, 32, val, 32)
#define REQ_QPC_SET_DST_IP(req, val)		QPC_SET(req, 1, 0, val, 32)
#define REQ_QPC_SET_SRC_IP(req, val)		QPC_SET(req, 1, 32, val, 32)
#define REQ_QPC_SET_DST_MAC_31_0(req, val)	QPC_SET(req, 2, 0, val, 32)
#define REQ_QPC_SET_DST_MAC_47_32(req, val)	QPC_SET(req, 2, 32, val, 16)
#define REQ_QPC_SET_SQ_NUM(req, val)		QPC_SET(req, 3, 24, val, 8)
#define REQ_QPC_SET_TM_GRANULARITY(req, val)	QPC_SET(req, 3, 56, val, 7)
#define REQ_QPC_SET_SOB_EN(req, val)		QPC_SET(req, 3, 63, val, 1)
#define REQ_QPC_SET_CONG_WINDOW(req, val)	QPC_SET(req, 4, 44, val, 22)
#define REQ_QPC_SET_TRANSPORT_SERVICE(req, val)	QPC_SET(req, 5, 49, val, 1)
#define REQ_QPC_SET_BURST_SIZE(req, val)	QPC_SET(req, 5, 50, val, 22)
#define REQ_QPC_SET_LAST_IDX(req, val)		QPC_SET(req, 6, 8, val, 22)
#define REQ_QPC_SET_SWQ_GRANULARITY(req, val)	QPC_SET(req, 7, 58, val, 1)
#define REQ_QPC_SET_WQ_BASE_ADDR(req, val)	QPC_SET(req, 7, 32, val, 24)
#define REQ_QPC_SET_SECURED(req, val)		QPC_SET(req, 7, 59, val, 2)
#define REQ_QPC_SET_VALID(req, val)		QPC_SET(req, 7, 63, val, 1)

#define REQ_QPC_GET_IN_WORK(req)		QPC_GET(req, 7, 61, 1)
#define REQ_QPC_GET_ERROR(req)			QPC_GET(req, 7, 62, 1)
#define REQ_QPC_GET_VALID(req)			QPC_GET(req, 7, 63, 1)

struct gaudi_qpc_responder {
	__le64	data[4];
};

#define RES_QPC_SET_DST_QP(res, val)		QPC_SET(res, 0, 0, val, 24)
#define RES_QPC_SET_PORT(res, val)		QPC_SET(res, 0, 24, val, 4)
#define RES_QPC_SET_PRIORITY(res, val)		QPC_SET(res, 0, 28, val, 2)
#define RES_QPC_SET_SQ_NUM(res, val)		QPC_SET(res, 2, 48, val, 8)
#define RES_QPC_SET_LKEY(res, val)		QPC_SET(res, 0, 32, val, 32)
#define RES_QPC_SET_DST_IP(res, val)		QPC_SET(res, 1, 0, val, 32)
#define RES_QPC_SET_SRC_IP(res, val)		QPC_SET(res, 1, 32, val, 32)
#define RES_QPC_SET_DST_MAC_31_0(res, val)	QPC_SET(res, 2, 0, val, 32)
#define RES_QPC_SET_DST_MAC_47_32(res, val)	QPC_SET(res, 2, 32, val, 16)
#define RES_QPC_SET_TRANSPORT_SERVICE(res, val)	QPC_SET(res, 2, 63, val, 1)
#define RES_QPC_SET_LOG_BUF_SIZE_MASK(res, val)	QPC_SET(res, 3, 24, val, 5)
#define RES_QPC_SET_SOB_EN(res, val)		QPC_SET(res, 3, 59, val, 1)
#define RES_QPC_SET_VALID(res, val)		QPC_SET(res, 3, 63, val, 1)
#define RES_QPC_SET_SECURED(res, val)		QPC_SET(res, 3, 60, val, 2)

#define RES_QPC_GET_CONN_STATE(res)		QPC_GET(res, 1, 30, 2)
#define RES_QPC_GET_IN_WORK(res)		QPC_GET(res, 3, 62, 1)
#define RES_QPC_GET_VALID(res)			QPC_GET(res, 3, 63, 1)

struct gaudi_sq_wqe {
	__le64	data[4];
};

#define CFG_SQ_WQE_OPCODE(swq, val) \
				((swq).data[0] |= cpu_to_le64((val) << 28))
#define CFG_SQ_WQE_LOCAL_ADDRESS_31_0(swq, val) \
				((swq).data[0] |= cpu_to_le64((val) << 32))
#define CFG_SQ_WQE_LOCAL_ADDRESS_49_32(swq, val) \
				((swq).data[1] |= cpu_to_le64((val)))
#define CFG_SQ_WQE_SIZE(swq, val) \
				((swq).data[1] |= cpu_to_le64((val) << 18))

/**
 * struct gaudi_en_core_info - Gaudi eth device core information.
 * @pcie_bar: PCIe bar virtual addresses.
 * @rx_mem_addr: array of Rx memory address of all ports.
 * @hbm_bar_cur_addr: pointer to current HBM bar address.
 * @tx_wq_base_addr: address of the Tx WQ.
 * @tx_ring_len: Tx cyclic buffer length (number of packets).
 * @rx_ring_len: Rx cyclic buffer length (number of packets).
 * @base_rx_irq: the base IRQ number of the Rx interrupt (for simulator use only).
 * @multi_msi_mode: whether we are working in multi MSI single MSI mode.
 *                  Multi MSI is possible only with IOMMU enabled.
 */
struct gaudi_en_core_info {
	void __iomem	*pcie_bar;
	void		**rx_mem_addr;
	u64		*hbm_bar_cur_addr;
	u64		tx_wq_base_addr;
	u32		tx_ring_len;
	u32		rx_ring_len;
	u32		base_rx_irq;
	u8		multi_msi_mode;
};

/**
 * struct gaudi_en_aux_ops - ASIC specific functions for core <-> eth drivers communication.
 * @read_mac_stat: read a MAC statistics value.
 * @read_tx_ci: read Tx cyclic buffer consumer index.
 * @read_tx_ci: read Rx cyclic buffer producer index.
 * @ring_tx_doorbell: ring the Tx door bell so the HW will send it.
 * @configure_rx_irq: configure the Rx IRQ.
 * @reenable_rx_irq: re-enable the Rx IRQ as it is masked upon a received packet.
 * @rx_irq_handler: handle an Rx interrupt.
 */
struct gaudi_en_aux_ops {
	/* eth2core */
	u64 (*read_mac_stat)(struct hl_aux_dev *aux_dev, u32 port, int idx, bool is_rx);
	u32 (*read_tx_ci)(struct hl_aux_dev *aux_dev, u32 port);
	u32 (*read_rx_pi)(struct hl_aux_dev *aux_dev, u32 port);
	void (*ring_tx_doorbell)(struct hl_aux_dev *aux_dev, u32 port, u32 pi);
	void (*configure_rx_irq)(struct hl_aux_dev *aux_dev, u32 port, bool enable);
	void (*reenable_rx_irq)(struct hl_aux_dev *aux_dev, u32 port);

	/* core2eth */
	irqreturn_t (*rx_irq_handler)(struct hl_aux_dev *aux_dev, u32 port);
};

#endif /* HL_GAUDI_H_ */
