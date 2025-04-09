def remove_nul_chars(input_filepath, output_filepath):
    with open(input_filepath, 'r', encoding='utf-8', errors='replace') as infile, \
         open(output_filepath, 'w', encoding='utf-8') as outfile:
        for line in infile:
            # 替换掉所有的 '\x00' 字符
            cleaned_line = line.replace('\x00', '')
            outfile.write(cleaned_line)

# 指定输入文件路径和输出文件路径
input_file_path = './vYHC-001.txt'
output_file_path = 'orign_eeg_data/vYHC-001_1.txt'

# 调用函数
remove_nul_chars(input_file_path, output_file_path)