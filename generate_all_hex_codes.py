# Generate all possible hex codes from 00 to FF in the 31E0 format

output_file = "all_hex_codes_31E0.txt"

with open(output_file, "w", encoding="utf-8") as outfile:
    line_nr = 0
    for hex_value in range(0x00, 0x100):  # 0 to 255 (0x00 to 0xFF)
        hex_str = f"{hex_value:02X}"
        line = f'{line_nr}: " I --- 29:162275 32:146231 --:------ 31E0 008 0000{hex_str}000100AA00"\n'
        line_nr += 1
        outfile.write(line)

print(f"Generated all 256 hex codes (00-FF) in file: {output_file}")
print("Sample lines:")
print('00: " I --- 29:162275 32:146231 --:------ 31E0 008 000000000100AA00"')
print('01: " I --- 29:162275 32:146231 --:------ 31E0 008 000001000100AA00"')
print('...')
print('FE: " I --- 29:162275 32:146231 --:------ 31E0 008 0000FE000100AA00"')
print('FF: " I --- 29:162275 32:146231 --:------ 31E0 008 0000FF000100AA00"')
