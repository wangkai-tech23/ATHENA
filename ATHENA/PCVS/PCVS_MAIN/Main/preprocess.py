import os
import re
import csv

output_folder = 'C:/Users/91237/Desktop/shiyan'
os.makedirs(output_folder, exist_ok=True)

input_file = 'C:/Users/91237/Desktop/shiyan/Time/correlated_signal_attack_1/correlated_signal_attack_1.log'  # 替换为你的数据集文件路径
with open(input_file, 'r') as f:
    lines = f.readlines()

time_series_dict = {}
for line in lines:
    parts = line.strip().split()
    if (len(parts) >= 3) and ('#' in parts[2]):
        time_str, _, id_data = parts
        time_str = re.sub(r'[^\d.]', '', time_str)
        time = round(float(time_str) - 1030000000, 6)
        id = id_data.split('#')[0]
        if id != "FFF":
            if id not in time_series_dict:
                time_series_dict[id] = []
            time_series_dict[id].append((time, id_data))

# for id, time_series in time_series_dict.items():
#     csv_filename = os.path.join(output_folder, f'{id}.csv')
#     with open(csv_filename, 'w') as csv_file:
#         csv_file.write('Time,ID,Data\n')
#         for time, id_data in time_series:
#             data = id_data.split('#')[1][-4:]  # 仅提取最后四位
#             try:
#                 decimal_data = int(data, 16)  # Convert hexadecimal to decimal
#             except ValueError:
#                 decimal_data = "Invalid"  # Handle cases where conversion might fail
#             csv_file.write(f'{time},{id},{decimal_data}\n')

summary_filename = os.path.join(output_folder, 'correlated_signal_attack_1.csv')
with open(summary_filename, 'w', newline='') as summary_file:
    writer = csv.writer(summary_file)
    header = list(time_series_dict.keys())
    writer.writerow(header)

    max_length = max(len(series) for series in time_series_dict.values())

    for i in range(max_length):
        row = []
        for id in header:
            if i < len(time_series_dict[id]):
                _, id_data = time_series_dict[id][i]
                data = id_data.split('#')[1][4:6]
                try:
                    decimal_data = int(data, 16)
                except ValueError:
                    decimal_data = "Invalid"
            else:
                decimal_data = ""
            row.append(decimal_data)
        writer.writerow(row)

print("CSV文件已保存在'实验'文件夹中")
