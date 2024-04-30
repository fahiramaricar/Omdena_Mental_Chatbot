""" This is to create smaller files from big data file(900 MB)"""
def split_large_file_size(input_file, chunk_size, output_prefix):
    with open(input_file, 'rb') as f:
        chunk_number = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            output_file = f"{output_prefix}_{chunk_number}.txt"
            with open(output_file, 'wb') as out:
                out.write(chunk)
            chunk_number += 1


input_file = 'tweetdata.txt'
chunk_size = 10 * 1024 * 1024  # 10 MB chunk size
output_prefix = 'twitter_chunk'  # Output file prefix
# split_large_file_size(input_file, chunk_size, output_prefix)

def split_large_file(input_file, lines_per_chunk, output_prefix):
    with open(input_file, 'r') as f:
        chunk_number = 0
        line_number = 0
        while True:
            chunk_lines = []
            for _ in range(lines_per_chunk):
                line = f.readline()
                if not line:
                    break
                chunk_lines.append(line)
                line_number += 1
            if not chunk_lines:
                break
            output_file = f"data_smaller_files/{output_prefix}_{chunk_number}.txt"
            with open(output_file, 'w') as out:
                out.writelines(chunk_lines)
            chunk_number += 1

# Example usage
input_file = 'tweetdata.txt'
lines_per_chunk = 1000  # Number of lines per chunk
output_prefix = 'twitter_chunk'  # Output file prefix
split_large_file(input_file, lines_per_chunk, output_prefix)

