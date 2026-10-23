// SPDX-License-Identifier: GPL-2.0

/* Copyright 2019 HabanaLabs, Ltd.
 * All Rights Reserved.
 */

#include "gaudi2P.h"
#include "gaudi2_nic.h"
#include "../include/gaudi2/asic_reg/gaudi2_regs.h"
#include <linux/module.h>
#include <linux/firmware.h>
#include <asm/unaligned.h>

#define NIC_PHY_CFG_SIZE \
		(mmNIC0_SERDES1_LANE0_REGISTER_0P00 - mmNIC0_SERDES0_LANE0_REGISTER_0P00)

#define NIC_PHY_CFG_BASE(port) \
		({ \
			u32 __port = port; \
			((u64)(NIC_MACRO_CFG_BASE(__port) + \
			NIC_PHY_CFG_SIZE * (u64)((__port) & 1))); \
		})

#define LANE_LO_OFF \
		(mmNIC0_SERDES0_LANE1_REGISTER_0P00 - mmNIC0_SERDES0_LANE0_REGISTER_0P00)

#define LANE_HI_OFF \
		(mmNIC0_SERDES0_LANE1_REGISTER_AI00 - mmNIC0_SERDES0_LANE0_REGISTER_AI00)

#define LANE_OFF(reg, lane) \
		({ \
			u32 __lane = lane; \
			((reg) < mmNIC0_SERDES0_LANE0_REGISTER_AI00) ? \
			((__lane) * LANE_LO_OFF) : ((__lane) * LANE_HI_OFF); \
		})

#define PHY_PRINT(port, lane, op, val, reg) \
	({ \
		if (hdev->nic.phy_regs_print) { \
			dev_info(hdev->dev, "[%s],Nic,%u,Port,%u,Lane,%d,%s,0x%08x,0x%08llx\n", \
				__func__, (port) >> 1, (port) & 0x1, lane, op, val, reg); \
			usleep_range(1000, 2000); \
		} \
	})

#define NIC_PHY_RREG32(reg) \
	({ \
		u32 _port = port; \
		u64 _reg = NIC_PHY_CFG_BASE(_port) + (reg); \
		u32 _val = RREG32(_reg); \
		PHY_PRINT(_port, -1, "read", _val, _reg); \
		_val; \
	})

#define NIC_PHY_WREG32(reg, val) \
	do { \
		u32 _port = port; \
		u64 _reg = NIC_PHY_CFG_BASE(_port) + (reg); \
		u32 _val = val; \
		WREG32(_reg, _val); \
		PHY_PRINT(_port, -1, "write", _val, _reg); \
	} while (0)

#define NIC_PHY_RMWREG32(reg, val, mask) \
	do { \
		u32 _port = port; \
		u64 _reg = NIC_PHY_CFG_BASE(_port) + (reg); \
		u32 _val = val; \
		u32 _mask = mask; \
		u32 _tmp = RREG32(_reg); \
		PHY_PRINT(_port, -1, "read(rmw)", _tmp, _reg); \
		_tmp &= ~_mask; \
		_tmp |= (_val << __ffs(_mask)); \
		WREG32(_reg, _tmp); \
		PHY_PRINT(_port, -1, "write(rmw)", _tmp, _reg); \
	} while (0)

#define NIC_PHY_RREG32_LANE(reg, lane) \
	({ \
		u32 _port = port; \
		u32 _lane = lane; \
		u64 _reg = reg; \
		u64 __reg = NIC_PHY_CFG_BASE(_port) + _reg + LANE_OFF(_reg, _lane); \
		u32 _val = RREG32(__reg); \
		PHY_PRINT(_port, _lane, "read", _val, __reg); \
		_val; \
	})

#define NIC_PHY_WREG32_LANE(reg, lane, val) \
	do { \
		u32 _port = port; \
		u32 _lane = lane; \
		u64 _reg = reg; \
		u64 __reg = NIC_PHY_CFG_BASE(_port) + _reg + LANE_OFF(_reg, _lane); \
		u32 _val = val; \
		WREG32(__reg, _val); \
		PHY_PRINT(_port, _lane, "write", _val, __reg); \
	} while (0)

#define NIC_PHY_RMWREG32_LANE(reg, lane, val, mask) \
	do { \
		u32 _port = port; \
		u32 _lane = lane; \
		u64 _reg = reg; \
		u64 __reg = NIC_PHY_CFG_BASE(_port) + _reg + LANE_OFF(_reg, _lane); \
		u32 _val = val; \
		u32 _mask = mask; \
		u32 _tmp = RREG32(__reg); \
		PHY_PRINT(_port, _lane, "read(rmw)", _tmp, __reg); \
		_tmp &= ~_mask; \
		_tmp |= (_val << __ffs(_mask)); \
		WREG32(__reg, _tmp); \
		PHY_PRINT(_port, _lane, "write(rmw)", _tmp, __reg); \
	} while (0)

#define NIC_PHY_READ_COUNTS_PER_MS		100000
#define NIC_PHY_FW_TIME_CONSTANT_RATIO		64
#define NIC_PHY_FW_TUNING_INTERVAL_MS		100
#define NIC_PHY_FW_TUNING_TIMEOUT_MS		60000 /* 1 minute */
#define NIC_PHY_POST_FW_TUNING_WAIT_MS		90000 /* 1.5 minutes */
#define NIC_PHY_CHECK_LINK_INTERVAL_MS		1000 /* 1 second */
#define NIC_PHY_PORT_FW_TUNED			0x3
#define NIC_PHY_PAM4_BER_FACTOR			53125000
#define NIC_PHY_NRZ_BER_FACTOR			25781250

#define NIC_PHY_TX_POL_MASK_HL225		0xF00000000430
#define NIC_PHY_RX_POL_MASK_HL225		0x0FFFFFFFFBCF
#define NIC_PHY_TX_POL_MASK_HLS2		0x0
#define NIC_PHY_RX_POL_MASK_HLS2		0x0
#define NIC_PHY_TX_POL_MASK_HL225_S		0xB0350203A3E2
#define NIC_PHY_RX_POL_MASK_HL225_S		0x404800002000

enum lane_state {
	READY,
	NOT_READY,
	FAILURE
};

static void soft_reset(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0x888,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0x0,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
}

static void logic_reset(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0x777,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0x0,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
}

static void cpu_reset(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0xAAA,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980D, 0x0,
				NIC0_SERDES0_REGISTER_980D_DOMAIN_RESET_MASK);
}

static int fw_cmd(struct hl_nic_port *nic_port, u32 cmd, u32 *detail, u32 expected_res,
		  u32 *res_ptr)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	u32 res, val, checks = 0;

	if (detail)
		NIC_PHY_WREG32(mmNIC0_SERDES0_REGISTER_9816, *detail);

	NIC_PHY_WREG32(mmNIC0_SERDES0_REGISTER_9815, cmd);

	do {
		usleep_range(1000, 2000);
		res = NIC_PHY_RREG32(mmNIC0_SERDES0_REGISTER_9815);
		if (checks++ > NIC_PHY_READ_COUNTS_PER_MS) {
			dev_dbg(hdev->dev, "timeout for PHY cmd 0x%x port %u\n", cmd, port);
			return -ETIMEDOUT;
		}
	} while (res == cmd);

	val = (res >> 8) & 0xF;

	if (val != expected_res) {
		dev_dbg(hdev->dev, "cmd 0x%x returned error 0x%x port %u\n",
			cmd, val, port);
		return -EFAULT;
	}

	*res_ptr = res;

	return 0;
}

static void clock_init(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	u32 first_val, second_val;

	if (port & 0x1) { /* raven 1 */
		if (lane == 0) {
			first_val = 0xA9E0;
			second_val = 0x9B9E;
		} else { /* lane 1 */
			first_val = 0xA9E0;
			second_val = 0x9B9E;
		}
	} else { /* raven 0 */
		if (lane == 0) {
			first_val = 0x59E0;
			second_val = 0x9B5E;
		} else { /* lane 1 */
			first_val = 0xA9E0;
			second_val = 0x9B9E;
		}
	}

	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PCC, lane, first_val);
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0NF3, lane, second_val);
}

static u32 int_to_twos(s32 val, u8 bitwidth)
{
	return val < 0 ? (1 << bitwidth) + val : val;
}

static void set_tx_taps(struct hl_nic_port *nic_port, int lane, s32 tx_pre2, s32 tx_pre1,
			s32 tx_main, s32 tx_post1, s32 tx_post2)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA5, lane, int_to_twos(tx_pre2, 8),
				NIC0_SERDES0_LANE0_REGISTER_0PA5_TX_PRE_2_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA7, lane, int_to_twos(tx_pre1, 8),
				NIC0_SERDES0_LANE0_REGISTER_0PA7_TX_PRE_1_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA9, lane, int_to_twos(tx_main, 8),
				NIC0_SERDES0_LANE0_REGISTER_0PA9_TX_MAIN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAB, lane, int_to_twos(tx_post1, 8),
				NIC0_SERDES0_LANE0_REGISTER_0PAB_TX_POST_1_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAD, lane, int_to_twos(tx_post2, 8),
				NIC0_SERDES0_LANE0_REGISTER_0PAD_TX_POST_2_MASK);

	dev_dbg(hdev->dev,
		"Card %u Port %u lane %d tx taps: pre2 %d, pre1 %d, main %d, post1 %d, post2 %d\n",
		card_location, port, lane, tx_pre2, tx_pre1, tx_main, tx_post1, tx_post2);
}

static void init_pam4(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	/* Tx initial */
	/* auto symmetric, scale */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, 0xFAC9);
	/* data prbs */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0xEF00);
	/* data, prbs */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0x4000);
	/* cursor -2 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA5, lane, 0x0100);
	/* cursor -1 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA7, lane, 0xF900);
	/* cursor -main */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA9, lane, 0x1700);
	/* cursor +1 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAB, lane, 0x0000);
	/* cursor +2 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAD, lane, 0x0000);

	/* Rx initial */
	/* ac-couple always */
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PF8, lane, 0x1,
			      NIC0_SERDES0_LANE0_REGISTER_0PF8_AC_COUPLE_EN_MASK);
}

static void set_lane_mode(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	if (pam4) {
		/* Disable NRZ mode */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0,
					NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_MODE_MASK);
		/* Disable NRZ PRBS Generator */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0,
					NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_GEN_EN_MASK);
		/* Enable PAM4 mode */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P41, lane, 1,
					NIC0_SERDES0_LANE0_REGISTER_0P41_PAM4_EN_MASK);
		/* Enable PAM4 PRBS Generator */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 1,
					NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PAM4_TEST_EN_MASK);
	} else {
		/* Disable PAM4 mode */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P41, lane, 0,
					NIC0_SERDES0_LANE0_REGISTER_0P41_PAM4_EN_MASK);
		/* Disable PAM4 PRBS Generator */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
					NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PAM4_TEST_EN_MASK);
		/* Enable NRZ mode */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 1,
					NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_MODE_MASK);
		/* Enable NRZ PRBS Generator */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 1,
					NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_GEN_EN_MASK);
	}
}

static void prbs_mode_select_no_prbs(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port, val;

	val = pam4 ? 0 : 1;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_CLK_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_GEN_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_MODE_MASK);

	if (pam4) {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0,
				      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_HALF_RATE_EN_MASK);

		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
				      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_TEST_DATA_SRC_MASK);
	}

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_CLK_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PAM4_TEST_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_GEN_EN_MASK);

	if (pam4) {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_PU_PRBS_CHKR_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_PU_PRBS_SYNC_CHKR_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_RX_PRBS_AUTO_SYNC_EN_MASK);
	}
}

static void prbs_mode_select(struct hl_nic_port *nic_port, int lane, bool pam4,
			     char *prbs_mode)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port, val;

	if (!prbs_mode || strncmp(prbs_mode, "PRBS", strlen("PRBS")))
		return prbs_mode_select_no_prbs(nic_port, lane, pam4);

	if (pam4) {
		if (!strncmp(prbs_mode, "PRBS9", strlen(prbs_mode)))
			val = 0;
		else if (!strncmp(prbs_mode, "PRBS13", strlen(prbs_mode)))
			val = 1;
		else if (!strncmp(prbs_mode, "PRBS15", strlen(prbs_mode)))
			val = 2;
		else /* PRBS31 */
			val = 3;

		/* Tx */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, val,
					NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_MODE_MASK);
		/* Rx */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, val,
					NIC0_SERDES0_LANE0_REGISTER_0P43_PRBS_MODE_SEL_MASK);
	} else {
		if (!strncmp(prbs_mode, "PRBS9", strlen(prbs_mode)))
			val = 0;
		else if (!strncmp(prbs_mode, "PRBS15", strlen(prbs_mode)))
			val = 1;
		else if (!strncmp(prbs_mode, "PRBS23", strlen(prbs_mode)))
			val = 2;
		else /* PRBS31 */
			val = 3;

		/* Tx */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, val,
					NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_MODE_MASK);
		/* Rx */
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N61, lane, val,
					NIC0_SERDES0_LANE0_REGISTER_0N61_NRZ_PRBS_MODE_SEL_MASK);
	}

	val = pam4 ? 0 : 1;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_CLK_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_PRBS_GEN_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, val,
			      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_NRZ_MODE_MASK);

	if (pam4)
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0,
				      NIC0_SERDES0_LANE0_REGISTER_0PB0_TX_HALF_RATE_EN_MASK);

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 1,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_TEST_DATA_SRC_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 1,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_CLK_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 1,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PAM4_TEST_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 1,
			      NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_GEN_EN_MASK);

	if (pam4) {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_PU_PRBS_CHKR_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_PU_PRBS_SYNC_CHKR_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0P43_RX_PRBS_AUTO_SYNC_EN_MASK);
	} else {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N61, lane, 1,
				      NIC0_SERDES0_LANE0_REGISTER_0N61_PRBS_CHKR_EN_MASK);
	}
}

static void set_pol(struct hl_nic_port *nic_port, int lane, u32 tx_pol, u32 rx_pol)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	/* Tx (common for both PAM4 and NRZ) */
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, tx_pol,
				NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_ANA_OUT_FLIP_MASK);

	/* Rx */
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, rx_pol,
				NIC0_SERDES0_LANE0_REGISTER_0P43_RX_DATA_FLIP_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N61, lane, rx_pol,
				NIC0_SERDES0_LANE0_REGISTER_0N61_PRBS_CHECK_FLIP_MASK);

	dev_dbg(hdev->dev, "Port %u lane %d polarity: tx %u rx %u\n", port, lane, tx_pol, rx_pol);
}

static void set_gc(struct hl_nic_port *nic_port, int lane, u32 tx_gc, u32 rx_gc)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, tx_gc,
				NIC0_SERDES0_LANE0_REGISTER_0PAF_TX_GRAYCODE_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P42, lane, rx_gc,
				NIC0_SERDES0_LANE0_REGISTER_0P42_RX_GRAYCODE_EN_MASK);
}

static void set_pc(struct hl_nic_port *nic_port, int lane, u32 tx_pc, u32 rx_pc)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, tx_pc,
				NIC0_SERDES0_LANE0_REGISTER_0PAF_TX_PRECODE_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P42, lane, rx_pc,
				NIC0_SERDES0_LANE0_REGISTER_0P42_RX_PRECODE_EN_MASK);
}

static void set_msblsb(struct hl_nic_port *nic_port, int lane, u32 tx_msblsb, u32 rx_msblsb)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, tx_msblsb,
				NIC0_SERDES0_LANE0_REGISTER_0PAF_TX_SWAP_MSB_LSB_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, rx_msblsb,
				NIC0_SERDES0_LANE0_REGISTER_0P43_RX_SWAP_MSB_LSB_MASK);
}

static void init_lane_for_fw(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	struct cpucp_nic_info *nic_info;
	u32 port = nic_port->port;
	u32 tx_pol, rx_pol;
	u32 tx_gc, rx_gc;
	u32 abs_lane_idx;

	nic_info = &hdev->asic_prop.cpucp_nic_info;

	abs_lane_idx = (port << 1) + lane;
	tx_pol = (le64_to_cpu(nic_info->pol_tx_mask[0]) >> abs_lane_idx) & 1;
	rx_pol = (le64_to_cpu(nic_info->pol_rx_mask[0]) >> abs_lane_idx) & 1;

	tx_gc = pam4 ? 1 : 0;
	rx_gc = tx_gc;

	set_lane_mode(nic_port, lane, pam4);

	/* TX Taps scales = (0.5,0.5,1,0.5,0.5) */
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, 0x4, 0x3E);

	/* The polairy values which are provided by F/W version < v1.4 are flipped,
	 * so need to set them correctly.
	 */
	if (hdev->nic.use_fw_serdes_info && (hdev->fw_major_version < 35)) {
		tx_pol = !tx_pol;
		rx_pol = !rx_pol;
	}

	set_pol(nic_port, lane, tx_pol, rx_pol);
	set_gc(nic_port, lane, tx_gc, rx_gc);
	set_pc(nic_port, lane, 0, 0);
	set_msblsb(nic_port, lane, 0, 0);

	prbs_mode_select(nic_port, lane, pam4, "PRBS31");
}

static void set_functional_mode(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
				NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_CLK_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
				NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PAM4_TEST_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0,
				NIC0_SERDES0_LANE0_REGISTER_0PA0_TX_PRBS_GEN_EN_MASK);

	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_AN10, lane, 0);

	dev_dbg(hdev->dev, "Card %u Port %u lane %d: Switched to functional mode\n",
		hdev->nic.card_location, port, lane);
}

static int fw_config_speed_pam4(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port, speed = 0x9;
	u32 abs_lane_idx, ignore;
	s32 *taps;
	int rc;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	/* clear go bit */
	NIC_PHY_WREG32(mmNIC0_SERDES0_REGISTER_980F, 0x800);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980F, 0x0, 0x8000);

	/* Disable AN/LT lane swapping */
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_AJ40, lane, 0x0,
				NIC0_SERDES0_LANE0_REGISTER_AJ40_ANLT_LANE_SWAPPING_EN_MASK);

	init_pam4(nic_port, lane);
	init_lane_for_fw(nic_port, lane, true);

	rc = fw_cmd(nic_port, 0x80D0 | lane, &speed, 0x8, &ignore);
	if (rc) {
		dev_err(hdev->dev,
			"Card %u Port %u lane %d: F/W config speed PAM4 failed, rc %d\n",
			card_location, port, lane, rc);
		return rc;
	}

	clock_init(nic_port, lane);

	abs_lane_idx = (port << 1) + lane;
	taps = hdev->nic.phy_tx_taps[abs_lane_idx].pam4_taps;
	set_tx_taps(nic_port, lane, taps[0], taps[1], taps[2], taps[3], taps[4]);

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PEA, lane, 0x60,
				NIC0_SERDES0_LANE0_REGISTER_0PEA_VDACCLKPHASE0_MASK);

	/* in case we want to show PHY BER statistics, we need to set functional mode only after
	 * F/W tuning done.
	 */
	if (!hdev->nic.phy_show_ber) {
		set_functional_mode(nic_port, lane);
		nic_port->phy_func_mode_en = true;
	}

	/* set go bit */
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980F, 0x1, 0x8000);

	return 0;
}

static int fw_config_vcocap(struct hl_nic_port *nic_port, int lane, u32 mode, u32 counter_value)
{
	u32 ignore;

	return fw_cmd(nic_port, 0x6000 | (mode << 4) | lane, &counter_value, 14, &ignore);
}

static int cal_pll_156MHz_ref(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	int rc;

	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x0,
				NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_32T_CLK_SEL_MASK);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x0,
				NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_32T_CLK_SEL_MASK);

	switch (nic_port->data_rate) {
	case NIC_DR_50:
	case NIC_DR_26:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x5);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x4FFA);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x40);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x0);
		if (rc)
			return rc;

		break;
	case NIC_DR_25:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x5);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x27FA);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x40);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x0);
		if (rc)
			return rc;

		break;
	case NIC_DR_10:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x2);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x0FFD);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x40);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x0);
		if (rc)
			return rc;

		break;
	}

	usleep_range(500, 1000);

	if (lane == 0) {
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x1,
					NIC0_SERDES0_REGISTER_9825_PLL_LOCK_SRC_SEL_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x0,
					NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_EN_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x1,
					NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_EN_MASK);
	} else {
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x2,
					NIC0_SERDES0_REGISTER_9825_PLL_LOCK_SRC_SEL_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x0,
					NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_EN_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x1,
					NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_EN_MASK);
	}

	usleep_range(500, 1000);

	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x1,
				NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_32T_CLK_SEL_MASK);
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x1,
				NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_32T_CLK_SEL_MASK);

	switch (nic_port->data_rate) {
	case NIC_DR_50:
	case NIC_DR_26:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x5);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x4FFA);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x30);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x1);
		if (rc)
			return rc;

		break;
	case NIC_DR_25:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x5);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x27FA);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x30);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x1);
		if (rc)
			return rc;

		break;
	case NIC_DR_10:
		rc = fw_config_vcocap(nic_port, lane, 1, 0x2);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 2, 0x0FFD);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 3, 0x40);
		if (rc)
			return rc;

		rc = fw_config_vcocap(nic_port, lane, 4, 0x1);
		if (rc)
			return rc;

		break;
	}

	usleep_range(500, 1000);

	if (lane == 0) {
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x1,
					NIC0_SERDES0_REGISTER_9825_PLL_LOCK_SRC_SEL_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x0,
					NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_EN_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x1,
					NIC0_SERDES0_REGISTER_9825_PLL_0_LOCK_EN_MASK);
	} else {
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_9825, 0x2,
					NIC0_SERDES0_REGISTER_9825_PLL_LOCK_SRC_SEL_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x0,
					NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_EN_MASK);
		NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_982E, 0x1,
					NIC0_SERDES0_REGISTER_982E_PLL_1_LOCK_EN_MASK);
	}

	usleep_range(500, 1000);

	return 0;
}

static void init_nrz(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	/* Tx initial */
	/* auto symmetric, scale */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAF, lane, 0xF8C9);
	/* data prbs */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA0, lane, 0xE320);
	/* data, prbs */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PB0, lane, 0x4820);
	/* cursor -2 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA5, lane, 0x0000);
	/* cursor -1 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA7, lane, 0xFC00);
	/* cursor -main */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PA9, lane, 0x1800);
	/* cursor +1 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAB, lane, 0x0000);
	/* cursor +2 */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PAD, lane, 0x0000);

	/* Rx initial */
	/* ac-couple always */
	NIC_PHY_WREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0PF8, lane, 0xEC06);
}

static int fw_config_speed_nrz(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port, speed = 0x3;
	u32 abs_lane_idx, ignore;
	s32 *taps;
	int rc;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	/* clear go bit */
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980F, 0x0, 0x8000);

	rc = fw_cmd(nic_port, 0x80C0 | lane, &speed, 0x8, &ignore);
	if (rc) {
		dev_err(hdev->dev,
			"Card %u Port %u lane %d: F/W config speed NRZ failed, rc %d\n",
			card_location, port, lane, rc);
		return rc;
	}

	init_nrz(nic_port, lane);

	abs_lane_idx = (port << 1) + lane;
	taps = hdev->nic.phy_tx_taps[abs_lane_idx].nrz_taps;
	set_tx_taps(nic_port, lane, taps[0], taps[1], taps[2], taps[3], taps[4]);

	clock_init(nic_port, lane);
	init_lane_for_fw(nic_port, lane, false);

	/* in case we want to show PHY BER statistics, we need to set functional mode only after
	 * F/W tuning done.
	 */
	if (!hdev->nic.phy_show_ber) {
		set_functional_mode(nic_port, lane);
		nic_port->phy_func_mode_en = true;
	}

	/* set go bit */
	NIC_PHY_RMWREG32(mmNIC0_SERDES0_REGISTER_980F, 0x1, 0x8000);

	return 0;
}

static void lane_reset_nrz(struct hl_nic_port *nic_port, int lane)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P41, lane, 0x1,
				NIC0_SERDES0_LANE0_REGISTER_0P41_PAM4_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P00, lane, 0x1,
				NIC0_SERDES0_LANE0_REGISTER_0P00_PAM4_SM_RESET_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P00, lane, 0x0,
				NIC0_SERDES0_LANE0_REGISTER_0P00_PAM4_SM_RESET_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P41, lane, 0x0,
				NIC0_SERDES0_LANE0_REGISTER_0P41_PAM4_EN_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N81, lane, 0x1,
				NIC0_SERDES0_LANE0_REGISTER_0N81_NRZ_SM_RESET_MASK);
	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N81, lane, 0x0,
				NIC0_SERDES0_LANE0_REGISTER_0N81_NRZ_SM_RESET_MASK);
}

static int fw_config(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	int rc;

	rc = cal_pll_156MHz_ref(nic_port, lane);
	if (rc) {
		dev_err(hdev->dev, "Port %u lane %d: PLL calibration failed, rc %d\n",
			port, lane, rc);
		return rc;
	}

	if (pam4) {
		rc = fw_config_speed_pam4(nic_port, lane);
		if (rc) {
			dev_err(hdev->dev, "Port %u lane %d: F/W PAM4 config failed, rc %d\n",
				port, lane, rc);
			return rc;
		}
	} else {
		lane_reset_nrz(nic_port, lane);
		rc = fw_config_speed_nrz(nic_port, lane);
		if (rc) {
			dev_err(hdev->dev, "Port %u lane %d: F/W NRZ config failed, rc %d\n",
				port, lane, rc);
			return rc;
		}
	}

	return 0;
}

static void phy_port_reset(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	int lane;

	dev_dbg(hdev->dev, "Port %u: reset PHY\n", nic_port->port);

	soft_reset(nic_port);
	usleep_range(500, 1000);

	for (lane = 0 ; lane < 2 ; lane++)
		clock_init(nic_port, lane);

	cpu_reset(nic_port);
	logic_reset(nic_port);

	usleep_range(500, 1000);
}

static void prbs_reset(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				NIC0_SERDES0_LANE0_REGISTER_0P43_RX_PRBS_AUTO_SYNC_EN_MASK);

	if (pam4) {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 1,
				NIC0_SERDES0_LANE0_REGISTER_0P43_PRBS_SYNC_CNTR_RESET_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 0,
				NIC0_SERDES0_LANE0_REGISTER_0P43_PRBS_SYNC_CNTR_RESET_MASK);
	} else {
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N61, lane, 1,
					NIC0_SERDES0_LANE0_REGISTER_0N61_PRBS_CNTR_RESET_MASK);
		NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N61, lane, 0,
					NIC0_SERDES0_LANE0_REGISTER_0N61_PRBS_CNTR_RESET_MASK);
	}

	NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P43, lane, 0,
				NIC0_SERDES0_LANE0_REGISTER_0P43_RX_PRBS_AUTO_SYNC_EN_MASK);
}

static u64 _get_prbs_cnt(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	u64 cnt;

	if (pam4)
		cnt = (((u64) NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P50, lane)) << 16)
			+ NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P51, lane);
	else
		cnt = (((u64) NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N66, lane)) << 16)
			+ NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N67, lane);

	return cnt;
}

static enum lane_state get_prbs_cnt(struct hl_nic_port *nic_port, int lane, bool pam4,
				    u64 prbs_prev_cnt, u64 *prbs_new_cnt)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port, phy_ready;
	u64 cnt;

	if (pam4) {
		phy_ready = (NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0P6A, lane) &
				NIC0_SERDES0_LANE0_REGISTER_0P6A_RX_READ_PHY_READY_MASK) >>
				NIC0_SERDES0_LANE0_REGISTER_0P6A_RX_READ_PHY_READY_SHIFT;
	} else {
		phy_ready = (NIC_PHY_RREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0N2E, lane) &
				NIC0_SERDES0_LANE0_REGISTER_0N2E_NRZ_READ_PHY_READY_MASK) >>
				NIC0_SERDES0_LANE0_REGISTER_0N2E_NRZ_READ_PHY_READY_SHIFT;
	}

	if (!phy_ready)
		return NOT_READY;

	cnt = _get_prbs_cnt(nic_port, lane, pam4);

	/* check PRBS counter wrapped around */
	if (cnt < prbs_prev_cnt) {
		if ((prbs_prev_cnt - cnt) < 0x10000)
			return FAILURE;

		cnt = _get_prbs_cnt(nic_port, lane, pam4);
	}

	*prbs_new_cnt = cnt;

	return READY;
}

static void _print_ber(struct hl_nic_port *nic_port, int lane, u64 total_cnt, u64 error_cnt)
{
	struct hl_device *hdev = nic_port->hdev;
	u64 total_high_digits, error_high_digits, integer, frac;
	u32 card_location, port;
	u8 total_num_digits, error_num_digits, exp;
	int i;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	total_num_digits = hl_nic_get_num_of_digits(total_cnt);
	error_num_digits = hl_nic_get_num_of_digits(error_cnt);

	if (total_num_digits > 2) {
		total_high_digits = total_cnt;

		for (i = 0 ; i < total_num_digits - 2 ; i++)
			total_high_digits = total_high_digits / 10;
	} else {
		total_high_digits = total_cnt;
	}

	if (!total_high_digits)
		return;

	if (error_num_digits > 2) {
		error_high_digits = error_cnt;

		for (i = 0 ; i < error_num_digits - 2 ; i++)
			error_high_digits = error_high_digits / 10;
	} else {
		error_high_digits = error_cnt;
	}

	exp = total_num_digits - error_num_digits;

	if (error_high_digits < total_high_digits) {
		error_high_digits *= 10;
		exp++;
	}

	integer = div_u64(error_high_digits, total_high_digits);
	frac = div_u64(((error_high_digits - (integer * total_high_digits)) * 10),
			total_high_digits);

	dev_info(hdev->dev,
		"Card %u Port %u lane %d: total_cnt %llu error_cnt %llu BER %llu.%llue-%u\n",
		card_location, port, lane, total_cnt, error_cnt, integer, frac, exp);
}

static void print_ber(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	u64 prbs_err_cnt_pre, prbs_prev_cnt, prbs_err_cnt_post, prbs_err_cnt,
	    prbs_reset_time_jiffies, prbs_accum_time_jiffies, prbs_accum_time_ms,
	    factor, error_cnt, total_cnt;
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port;
	enum lane_state state;
	u8 iter_num = 50;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	prbs_reset(nic_port, lane, pam4);
	prbs_reset_time_jiffies = jiffies;
	prbs_err_cnt_pre = _get_prbs_cnt(nic_port, lane, pam4);

	prbs_prev_cnt = prbs_err_cnt_pre;

	while (true) {
		msleep(500);

		state = get_prbs_cnt(nic_port, lane, pam4, prbs_prev_cnt, &prbs_err_cnt_post);
		prbs_accum_time_jiffies = jiffies - prbs_reset_time_jiffies;
		prbs_accum_time_ms = jiffies_to_msecs(prbs_accum_time_jiffies);
		prbs_err_cnt = prbs_err_cnt_post - prbs_err_cnt_pre;

		if (state != READY) {
			dev_err(hdev->dev, "Card %u Port %u lane %d: No BER (state = %s)\n",
				card_location, port, lane,
				(state == NOT_READY) ? "NOT_READY" : "FAILURE");
			return;
		}

		if (prbs_accum_time_ms >= 5000 || prbs_err_cnt >= 10000000)
			break;

		prbs_prev_cnt = prbs_err_cnt_post;
		iter_num--;
	}

	factor = pam4 ? NIC_PHY_PAM4_BER_FACTOR : NIC_PHY_NRZ_BER_FACTOR;

	dev_info(hdev->dev, "Card %u Port %u lane %d: measurement duration - %llu ms\n",
		card_location, port, lane, prbs_accum_time_ms);

	error_cnt = prbs_err_cnt;
	total_cnt = prbs_accum_time_ms * factor;

	_print_ber(nic_port, lane, total_cnt, error_cnt);
}

static void check_pcs_link(struct hl_nic_port *nic_port)
{
	bool link_up;

	if (nic_port->pcs_link == nic_port->prev_pcs_link)
		return;

	link_up = nic_port->pcs_link ? true : false;
	hl_nic_phy_set_port_status(nic_port, link_up);
	nic_port->prev_pcs_link = nic_port->pcs_link;
}

static u32 rv_debug(struct hl_nic_port *nic_port, int lane, u32 mode, u32 index)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port, cmd, res;

	cmd = 0xB000 + ((mode & 0xF) << 4) + lane;

	fw_cmd(nic_port, cmd, &index, 0xB, &res);

	return NIC_PHY_RREG32(mmNIC0_SERDES0_REGISTER_9816);
}

static int fw_tuning(struct hl_nic_port *nic_port, int lane, bool pam4)
{
	u32 exp_state, state, mode;

	if (pam4) {
		exp_state = 0x8F00;
		mode = 2;
	} else {
		exp_state = 0x9A00;
		mode = 1;
	}

	state = rv_debug(nic_port, lane, mode, 0);

	if (((u16)state) != exp_state)
		return -EAGAIN;

	return 0;
}

static void do_fw_tuning(struct hl_nic_port *nic_port)
{
	struct gaudi2_nic_port *gaudi2_nic = nic_port->nic_specific;
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port;
	int lane, rc;
	bool pam4;

	card_location = hdev->nic.card_location;
	port = nic_port->port;
	pam4 = (nic_port->data_rate == NIC_DR_50);

	for (lane = 0 ; lane < 2 ; lane++) {
		if (gaudi2_nic->phy_fw_tuned_bitmap & BIT(lane))
			continue;

		rc = fw_tuning(nic_port, lane, pam4);
		if (rc) {
			if (ktime_after(ktime_get(), nic_port->fw_tuning_limit_ts)) {
				dev_dbg(hdev->dev, "Card %u Port %u lane %d: F/W tuning limit\n",
					card_location, port, lane);

				hl_nic_phy_port_reconfig(nic_port);
				return;
			}
		} else {
			gaudi2_nic->phy_fw_tuned_bitmap |= BIT(lane);
		}
	}

	if (gaudi2_nic->phy_fw_tuned_bitmap == NIC_PHY_PORT_FW_TUNED) {
		nic_port->phy_fw_tuned = true;

		dev_dbg(hdev->dev, "Card %u Port %u: F/W tuning passed\n", card_location, port);

		if (hdev->nic.phy_show_ber) {
			dev_dbg(hdev->dev,
				"Card %u Port %u: Waiting %d seconds before switching to functional mode\n",
				card_location, port, NIC_PHY_POST_FW_TUNING_WAIT_MS / 1000);

			/* Wait 5 seconds before calculating BER */
			msleep(5000);

			for (lane = 0; lane < 2 ; lane++)
				print_ber(nic_port, lane, pam4);
		}
	}
}

static void gaudi2_nic_phy_link_status_work_old(struct work_struct *work)
{
	struct hl_nic_port *nic_port;
	u32 timeout_ms;

	nic_port = container_of(work, struct hl_nic_port, link_status_work.work);

	if (nic_port->phy_fw_tuned) {
		if (nic_port->phy_func_mode_en) {
			check_pcs_link(nic_port);
		} else {
			set_functional_mode(nic_port, 0);
			set_functional_mode(nic_port, 1);

			nic_port->phy_func_mode_en = true;
		}
	} else {
		do_fw_tuning(nic_port);
	}

	if (!nic_port->phy_fw_tuned)
		timeout_ms = NIC_PHY_FW_TUNING_INTERVAL_MS;
	else if (!nic_port->phy_func_mode_en)
		timeout_ms = NIC_PHY_POST_FW_TUNING_WAIT_MS;
	else
		timeout_ms = NIC_PHY_CHECK_LINK_INTERVAL_MS;

	queue_delayed_work(nic_port->wq, &nic_port->link_status_work,
			msecs_to_jiffies(timeout_ms));
}

static void gaudi2_nic_phy_port_start_stop_old(struct hl_nic_port *nic_port, bool is_start)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 port = nic_port->port;
	int lane;

	for (lane = 0 ; lane < 2 ; lane++) {
		if (is_start) {
			/* Enable TX driver in SerDes */
			NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0NF8, lane, 1,
						NIC0_SERDES0_LANE0_REGISTER_0NF8_PU_VDRV_MASK);
			/* Enable F/W Rx tuning is done during power up flow */
		} else {
			/* Disable TX driver in SerDes */
			NIC_PHY_RMWREG32_LANE(mmNIC0_SERDES0_LANE0_REGISTER_0NF8, lane, 0,
						NIC0_SERDES0_LANE0_REGISTER_0NF8_PU_VDRV_MASK);
			/* Silence F/W Rx tuning */
			NIC_PHY_WREG32(mmNIC0_SERDES0_REGISTER_9815, 0x9000|lane);
		}
	}
}

static int gaudi2_nic_phy_port_power_up_old(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 data_rate = nic_port->data_rate;
	u32 port = nic_port->port;
	int rc, lane;

	phy_port_reset(nic_port);
	nic_port->phy_func_mode_en = false;

	for (lane = 0 ; lane < 2 ; lane++) {
		rc = fw_config(nic_port, lane, data_rate == NIC_DR_50);
		if (rc) {
			dev_err(hdev->dev,
				"Port %u lane %u: F/W config failed, rc %d\n", port, lane, rc);
			return rc;
		}
	}

	nic_port->fw_tuning_limit_ts = ktime_add_ms(ktime_get(), NIC_PHY_FW_TUNING_TIMEOUT_MS);

	return 0;
}

static void gaudi2_nic_phy_port_reconfig_old(struct hl_nic_port *nic_port)
{
	struct hl_device *hdev = nic_port->hdev;
	u32 card_location, port;
	int rc;

	if (!hdev->nic.phy_config_fw)
		return;

	card_location = hdev->nic.card_location;
	port = nic_port->port;

	dev_dbg(hdev->dev, "Card %u Port %u: reconfiguring PHY\n", card_location, port);

	rc = gaudi2_nic_phy_port_power_up_old(nic_port);
	if (rc)
		dev_err(hdev->dev, "Card %u Port %u: PHY reconfig failed\n", card_location, port);
}

void gaudi2_nic_phy_override_funcs_old(struct hl_device *hdev)
{
	struct hl_nic_port_funcs *port_funcs;
	struct hl_nic_port *nic_port;
	u32 port;

	port_funcs = hdev->asic_funcs->nic_funcs->port_funcs;

	port_funcs->phy_port_start_stop = gaudi2_nic_phy_port_start_stop_old;
	port_funcs->phy_port_power_up = gaudi2_nic_phy_port_power_up_old;
	port_funcs->phy_port_reconfig = gaudi2_nic_phy_port_reconfig_old;

	/* Go over all the active ports and override the link status work with the old one */
	for (port = 0 ; port < NIC_NUMBER_OF_PORTS ; port++) {
		if (!(hdev->nic_ports_mask & BIT(port)))
			continue;

		nic_port = &hdev->nic.nic_ports[port];
		INIT_DELAYED_WORK(&nic_port->link_status_work,
					gaudi2_nic_phy_link_status_work_old);
	}
}
