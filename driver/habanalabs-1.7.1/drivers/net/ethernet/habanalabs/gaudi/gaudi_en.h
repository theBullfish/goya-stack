/* SPDX-License-Identifier: GPL-2.0
 *
 * Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 *
 */

#ifndef GAUDI_EN_H_
#define GAUDI_EN_H_

#include "../common/habanalabs_en.h"
#include <linux/habanalabs/gaudi.h>

/**
 * struct gaudi_device - Gaudi device structure.
 * @ports: array of Gaudi ports structures.
 * @core_info: relevant info from the core device.
 * @aux_ops: pointer functions for core <-> eth drivers communication.
 * @pcie_bar: PCIe bar virtual addresses.
 * @hbm_bar_cur_addr: pointer to current HBM bar address.
 * @tx_wq_base_addr: address of the Tx WQ.
 * @tx_queue_len: Tx queue length (number of packets).
 * @rx_mem_len: Rx memory length (number of packets).
 * @base_rx_irq: the base IRQ number of the Rx interrupt (for simulator use only).
 * @multi_msi_mode: whether we are working in multi MSI single MSI mode.
 *                  Multi MSI is possible only with IOMMU enabled.
 */
struct gaudi_device {
	struct gaudi_port		*ports;
	struct gaudi_en_core_info	*core_info;
	struct gaudi_en_aux_ops		*aux_ops;
	void __iomem			*pcie_bar;
	u64				*hbm_bar_cur_addr;
	u64				tx_wq_base_addr;
	u32				tx_queue_len;
	u32				rx_mem_len;
	u32				base_rx_irq;
	u8				multi_msi_mode;
};

/**
 * struct gaudi_port - Gaudi port structure.
 * @hdev: habanalabs device structure.
 * @rx_mem_addr: Rx memory address.
 * @idx: port index.
 * @rx_pi: incremented by H/W for each received packet from the H/W.
 * @rx_ci: incremented by S/W for each received packet from the H/W.
 * @tx_pi: incremented by S/W for each sent packet to the H/W.
 * @tx_ci: incremented by H/W for each sent packet to the H/W.
 */
struct gaudi_port {
	struct hl_en_device	*hdev;
	void			*rx_mem_addr;
	u32			idx;
	u32			rx_pi;
	u32			rx_ci;
	u32			tx_pi;
	u32			tx_ci;
};

void gaudi_en_dcbnl_get_pfc_cnts(struct hl_en_port *port, void *ptr);

#endif /* GAUDI_EN_H_ */
