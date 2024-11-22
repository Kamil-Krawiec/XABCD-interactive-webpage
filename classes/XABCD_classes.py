from dataclasses import dataclass
from typing import Union, List, Optional


@dataclass
class XABCDPattern:
    XA_AB: Optional[Union[float, List[float]]]
    AB_BC: Optional[Union[float, List[float]]]
    BC_CD: Optional[Union[float, List[float]]]
    XA_AD: Optional[Union[float, List[float]]]
    name: str


@dataclass
class XABCDPatternFound:
    X: int
    A: int
    B: int
    C: int
    D: int  # Index of the last point in the pattern, entry is on the close of D
    error: float  # Error found
    name: str
    bull: bool


class ValidPatterns:
    GARTLEY = XABCDPattern(0.618, [0.382, 0.886], [1.13, 1.618], 0.786, "Gartley")
    BAT = XABCDPattern([0.382, 0.50], [0.382, 0.886], [1.618, 2.618], 0.886, "Bat")
    BUTTERFLY = XABCDPattern(0.786, [0.382, 0.886], [1.618, 2.24], [1.27, 1.41], "Butterfly")
    CRAB = XABCDPattern([0.382, 0.618], [0.382, 0.886], [2.618, 3.618], 1.618, "Crab")
    DEEP_CRAB = XABCDPattern(0.886, [0.382, 0.886], [2.0, 3.618], 1.618, "Deep Crab")
    CYPHER = XABCDPattern([0.382, 0.618], [1.13, 1.41], [1.27, 2.00], 0.786, "Cypher")
    SHARK = XABCDPattern(None, [1.13, 1.618], [1.618, 2.24], [0.886, 1.13], "Shark")
    ALL_PATTERNS = [GARTLEY, BAT, BUTTERFLY, CRAB, DEEP_CRAB, CYPHER, SHARK]
