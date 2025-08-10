import re
import matplotlib.pyplot as plt
import numpy as np
# deleted this one from matching.txt:
#72.5% -> 19: " I --- 29:162275 32:146231 --:------ 31E0 008 0000C8000100AA00"
# it seemed to be an outlier.
def percent_to_hex4(pct: float) -> str:
    """
    Convert a percentage to a 4-digit hex string using the linear fit,
    rounding to the nearest xx00 (last two digits always 00).
    """
    integer = int(round(coeffs[0] * pct + coeffs[1]))
    # Round to nearest multiple of 0x100 (256)
    integer = int(round(integer / 0x100) * 0x100)
    integer = max(0, min(integer, 0xFFFF))
    return f"{integer:04X}"

def percent_to_line(pct: float) -> str:
    """
    Generate a CAN log line like:
    " I --- 29:162275 32:146231 --:------ 31E0 008 0000XXXX0100AA00"
    where XXXX is the 4-digit hex from the linear fit for the percentage.
    """
    hex4 = percent_to_hex4(pct)
    # Compose the payload: 0000 + hex4 + 0100AA00
    payload = f"0000{hex4}0100AA00"
    return f" I --- 29:181232 32:146231 --:------ 31E0 008 {payload}"

results = []

with open("matching.txt", "r", encoding="utf-8") as infile:
    for line in infile:
        # Extract percentage as float
        pct_match = re.match(r'([\d.]+)%', line)
        pct = float(pct_match.group(1)) if pct_match else None

        # Extract the hex payload (last group of hex digits)
        payload_match = re.search(r'31E0 008 ([0-9A-Fa-f]+)', line)
        payload = payload_match.group(1) if payload_match else ""

        # Get hexadecimal numbers 5 to 8 (zero-based index 4:8)
        hex_5_8 = payload[4:8]
        hex_int = int(hex_5_8, 16) if len(hex_5_8) == 4 else None

        results.append((pct, hex_5_8, hex_int))

# Prepare data for plotting and fitting
x = [pct for pct, _, hex_int in results if pct is not None and hex_int is not None]
y = [hex_int for _, _, hex_int in results if hex_int is not None]

# Linear fit
coeffs = np.polyfit(x, y, 1)  # coeffs[0] = slope, coeffs[1] = intercept

print(f"Linear fit coefficients: {coeffs}")
fit_fn = np.poly1d(coeffs)
percentages = np.arange(0, 89.5+1e-6, 0.5)
for i in percentages:
    print(f"{i}: \"{percent_to_line(i)}\"")

plt.plot(x, y, 'o', label='Data')
plt.plot(x, fit_fn(x), '-', label=f'Fit: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}')
plt.xlabel('Percentage')
plt.ylabel('Hex 5-8 as Integer')
plt.title('Percentage vs Hexadecimal Value (5-8)')
plt.legend()
plt.grid(True)
plt.show()

print(f"Linear fit: integer = {coeffs[0]:.4f} * percentage + {coeffs[1]:.4f}")



# Example

