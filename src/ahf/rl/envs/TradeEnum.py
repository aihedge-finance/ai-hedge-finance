from enum import Enum


class TradeAction(str, Enum):
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"
    SHORT = "SHORT"
    COVER = "COVER"
    BUY_FEE = "BUY_FEE"
    CASH_CHANGE = "CASH_CHANGE"
    TRADE_CASH_CHANGE = "TRADE_CASH_CHANGE"
    ADJUST = "ADJUST"


class ErrorCodes(dict, Enum):
    """錯誤的編碼和錯誤訊息"""

    # ---------- TradeHold related ----------
    ERROR_TH000 = {"code": "7000", "sys_msg": "Trade put to HOLD", "message": "Trade put to HOLD"}
    ERROR_TH001 = {"code": "7001", "sys_msg": "buy_num cannot be less than min_qty", "message": "qty must be larger than min_qty"}
    ERROR_TH002 = {"code": "7002", "sys_msg": "sell_num cannot be less than min_qty", "message": "qty must be larger than min_qty"}
    ERROR_TH003 = {"code": "7003", "sys_msg": "buy_num * price must be larger than min_notional", "message": "amount must be larger than min_notion"}
    ERROR_TH004 = {"code": "7004", "sys_msg": "sell_num * price must be larger than min_notional", "message": "amount must be larger than min_notion"}
    ERROR_TH005 = {"code": "7005", "sys_msg": "not enough cash", "message": "not enough cash"}
    ERROR_TH006 = {"code": "7006", "sys_msg": "Short amount must be at least twice of min_notional", "message": "Short amount must be at least twice of min_notional"}
    ERROR_TH007 = {"code": "7007", "sys_msg": "margin_level cannot be less than margin_call_level", "message": "Margin level already too low"}
    ERROR_TH008 = {"code": "7008", "sys_msg": "safe_max_notional cannot be less than self.min_notional * d(2)", "message": "Loanable amount too low"}
    ERROR_TH009 = {"code": "7009", "sys_msg": "orderId is None", "message": "Order failed due to Exchange error"}
    ERROR_TH010 = {"code": "7010", "sys_msg": "Adjust position", "message": "Manual position due to system error"}
    ERROR_TH011 = {"code": "7011", "sys_msg": "asset*1.1 value is already more than trade cash", "message": "Amount in hand is about or more than trade cash, increase trade cash limitation if you want to buy more."}
    ERROR_TH012 = {"code": "7012", "sys_msg": "self.ds.silo.can_buy is False", "message": "You have to sell before you can buy again for this version of tradebot"}
    ERROR_TH013 = {"code": "7013", "sys_msg": "No asset to be sold", "message": "No asset to be sold"}


    ERROR_PM000 = {"code": "6000", "sys_msg": "Process status is not available", "message": "Process status is not available"}
    ERROR_PM001 = {"code": "6001", "sys_msg": "Process status is is ill-formatted", "message": "Process status is is ill-formatted"}
    ERROR_PM002 = {"code": "6002", "sys_msg": "Process cannot be started", "message": "Process cannot be started"}


