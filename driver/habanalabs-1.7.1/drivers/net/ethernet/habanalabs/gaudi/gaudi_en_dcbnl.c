// SPDX-License-Identifier: GPL-2.0

/* Copyright 2021-2022 HabanaLabs, Ltd.
 * All Rights Reserved.
 */

#include "gaudi_en.h"
#include "../common/habanalabs_en.h"

#define PFC_STAT_TX_OFFSET	17
#define PFC_STAT_RX_OFFSET	27

void gaudi_en_dcbnl_get_pfc_cnts(struct hl_en_port *port, void *ptr)
{
#ifdef CONFIG_DCB
	struct hl_en_device *hdev = port->hdev;
	struct gaudi_en_aux_ops *asic_aux_ops;
	struct hl_en_aux_ops *aux_ops;
	struct hl_aux_dev *aux_dev;
	struct ieee_pfc *pfc = ptr;
	u32 port_idx = port->idx;
	int i, tx_idx, rx_idx;

	aux_dev = hdev->aux_dev;
	aux_ops = aux_dev->aux_ops;
	asic_aux_ops = aux_ops->asic_ops;

	for (i = 0 ; i < HL_EN_PFC_PRIO_NUM ; i++) {
		tx_idx = PFC_STAT_TX_OFFSET + i;
		rx_idx = PFC_STAT_RX_OFFSET + i;

		pfc->requests[i] = asic_aux_ops->read_mac_stat(aux_dev, port_idx, tx_idx, false);
		pfc->indications[i] = asic_aux_ops->read_mac_stat(aux_dev, port_idx, rx_idx, true);
	}
#endif
}
