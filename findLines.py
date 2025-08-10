# Save as: extract_from_31E0.py

input_file = "logfile.txt"
output_file = "logfile_from_31E0.txt"

found = False

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if "31E0" in line:
            idx = line.find("31E0")
            if idx != -1:
                outfile.write(line[idx:])

print(f"Lines from first '31E0' onward saved to {output_file}")

input_file = "logfile_from_31E0.txt"
output_file = "logfile_from_31E0_nodupes.txt"

seen = set()
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if line not in seen:
            outfile.write(line)
            seen.add(line)

print(f"Duplicates removed. Output saved to {output_file}")

input_file = "logfile_from_31E0_nodupes.txt"
output_file = "logfile_from_31E0_sorted.txt"

def hex_key(line):
    # Remove spaces, keep only hex digits (0-9, A-F)
    hex_str = ''.join(c for c in line if c.isalnum())
    # Convert to int for sorting, fallback to 0 if not valid
    try:
        return int(hex_str, 16)
    except ValueError:
        return 0

with open(input_file, "r", encoding="utf-8") as infile:
    lines = infile.readlines()

# Sort lines by their hex value (spaces removed)
lines_sorted = sorted(lines, key=hex_key)

with open(output_file, "w", encoding="utf-8") as outfile:
    outfile.writelines(lines_sorted)

print(f"Sorted lines by hex value. Output saved to {output_file}")

input_file = "logfile_from_31E0_sorted.txt"
output_file = "logfile_medium_format.txt"

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for idx, line in enumerate(infile, 1):
        line = line.strip()
        # Compose the new line
        formatted = f'{idx}: " I --- 29:162275 32:146231 --:------ {line}"\n'
        outfile.write(formatted)

print(f"Formatted output saved to {output_file}")