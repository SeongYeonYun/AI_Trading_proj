from dataclasses import dataclass

@dataclass
class Cost:
    buy_fee: float = 0.00015
    sell_fee: float = 0.00015
    sell_tax: float = 0.0020
