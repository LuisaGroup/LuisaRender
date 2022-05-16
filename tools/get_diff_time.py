import os
import sys
import re
import csv


def get_diff_name(log_file_name: str):
    with open(log_file_name, 'r') as f:
        log_text = f.read()

    forward_time = re.findall('Rendering finished in ([0-9\.]*) ms', log_text)
    backward_time = re.findall('Backward propagation finished in ([0-9\.]*) ms', log_text)
    step_time = re.findall('Step finished in ([0-9\.]*) ms', log_text)

    output_path = os.path.splitext(log_file_name)[0] + '.csv'
    print(output_path)

    headers = ['time', 'time_forward', 'time_backward']
    data = []
    print(len(forward_time), len(backward_time), len(step_time))
    iter_num = len(backward_time)
    assert len(forward_time) == iter_num + 1
    assert len(step_time) == iter_num
    for i in range(iter_num):
        forward_time_i = float(forward_time[i])
        backward_time_i = float(backward_time[i]) + float(step_time[i])
        time_i = forward_time_i + backward_time_i
        data.append([
            time_i,
            forward_time_i,
            backward_time_i,
        ])

    with open(output_path, 'w', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(data)


if __name__ == '__main__':
    log_file_name = sys.argv[1]
    get_diff_name(log_file_name)
