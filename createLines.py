import numpy as np
# calibration table: list of (percentage, raw_value) pairs
CALIBRATION = [
    (46.5, 0x0E00),
    (47.0, 0x1200),
    (47.5, 0x1600),
    (48.5, 0x1A00),
    (49.0, 0x1E00),
    (49.5, 0x2000),
    (51.0, 0x2800),
    (51.5, 0x2C00),
    (53.5, 0x3600),
    (54.0, 0x3A00),
    (56.5, 0x4600),
    (59.0, 0x5600),
    (60.5, 0x5C00),
    (72.5, 0xC800),
]

def percent_to_raw(pct: float) -> int:
    # clamp
    if pct <= CALIBRATION[0][0]:
        return CALIBRATION[0][1]
    if pct >= CALIBRATION[-1][0]:
        return CALIBRATION[-1][1]

    # find bracketing entries and linearly interpolate
    for (p0, r0), (p1, r1) in zip(CALIBRATION, CALIBRATION[1:]):
        if p0 <= pct <= p1:
            frac = (pct - p0) / (p1 - p0)
            return int(round(r0 + (r1 - r0) * frac))

    raise ValueError(f"Percentage {pct} out of calibration range")

def raw_to_payload(raw: int) -> bytes:
    lo, hi = raw & 0xFF, (raw >> 8) & 0xFF
    # bytes: [0]=0x00, [1]=0x00, [2]=lo, [3]=hi, [4]=0x01, [5]=0x00, [6]=0xAA, [7]=0x00
    return bytes([0x00, 0x00, lo, hi, 0x01, 0x00, 0xAA, 0x00])

def percent_to_hex(pct: float) -> str:
    payload = raw_to_payload(percent_to_raw(pct))
    return "".join(f"{b:02X}" for b in payload)

def format_can_line(pct: float) -> str:
    """
    Recreate the log line:
    " I --- 29:162275 32:146231 --:------ 31E0 008 <payload>"
    """
    payload = percent_to_hex(pct)
    # note: leading space before I, fixed fields, then 31E0 008
    return f"\" I --- 29:162275 32:146231 --:------ 31E0 008 {payload}\""

if __name__ == "__main__":

    percentages = np.arange(44, 100.5, 0.5)
    for pct in percentages:
        print(f"{pct}: {format_can_line(pct)}")
