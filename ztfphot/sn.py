from dataclasses import dataclass


@dataclass
class SN:
    jd_first_epoch: float
    jd_min: float
    jd_max: float
    snname: str
    filename: str
