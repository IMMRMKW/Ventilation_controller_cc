# Read all lines containing "32:146231" from logfile.txt and save to a new file

input_file = "logfile.txt"
output_file = "lines_with_32_146231.txt"

try:
    with open(input_file, "r", encoding="utf-8") as infile:
        lines_with_target = []
        
        for line in infile:
            if "32:146231" in line:
                lines_with_target.append(line)
        
        print(f"Found {len(lines_with_target)} lines containing '32:146231'")
        
        # Write to output file
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.writelines(lines_with_target)
        
        print(f"Lines saved to {output_file}")

except FileNotFoundError:
    print(f"Error: {input_file} not found")
except Exception as e:
    print(f"Error: {e}")
