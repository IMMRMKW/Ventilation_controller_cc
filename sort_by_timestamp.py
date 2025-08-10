from datetime import datetime

# Read the file and sort lines by timestamp
input_file = "lines_with_32_146231.txt"
output_file = "lines_with_32_146231_sorted.txt"

try:
    with open(input_file, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    
    print(f"Read {len(lines)} lines from {input_file}")
    
    # Parse and sort lines by timestamp
    def extract_timestamp(line):
        try:
            # Extract timestamp from beginning of line (format: 2025-07-17T12:37:52.916224)
            timestamp_str = line.split()[0]
            return datetime.fromisoformat(timestamp_str)
        except (IndexError, ValueError):
            # Return a very old date for lines that can't be parsed
            return datetime.min
    
    # Sort lines by timestamp
    sorted_lines = sorted(lines, key=extract_timestamp)
    
    # Write sorted lines to output file
    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.writelines(sorted_lines)
    
    print(f"Sorted lines saved to {output_file}")
    
    # Show first and last timestamps to verify sorting
    if sorted_lines:
        first_timestamp = extract_timestamp(sorted_lines[0])
        last_timestamp = extract_timestamp(sorted_lines[-1])
        print(f"First timestamp: {first_timestamp}")
        print(f"Last timestamp: {last_timestamp}")

except FileNotFoundError:
    print(f"Error: {input_file} not found")
except Exception as e:
    print(f"Error: {e}")
