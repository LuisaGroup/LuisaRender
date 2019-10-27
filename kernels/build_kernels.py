from os import listdir, mkdir, system
from os.path import exists
from sys import argv

source_directory = argv[0][:argv[0].rfind("/") + 1]
build_directory = source_directory + "build/"
binary_directory = source_directory + "bin/"
include_directory = source_directory + "../src"

if not exists(build_directory):
    mkdir(build_directory)

if not exists(binary_directory):
    mkdir(binary_directory)


def source_file_path(name):
    return "{}{}.metal".format(source_directory, name)


def ir_file_path(name):
    return "{}{}.air".format(build_directory, name)


build_command_template = "xcrun -sdk macosx metal -std=macos-metal2.2 -Ofast -Wall -Wextra -ffast-math -c {src_path} -o {ir_path} -I {include_dir}"
kernel_names = [f[:-6] for f in listdir(source_directory) if f.endswith(".metal")]
for name in kernel_names:
    src_path = source_file_path(name)
    ir_path = ir_file_path(name)
    command = build_command_template.format(src_path=src_path, ir_path=ir_path, include_dir=include_directory)
    print("Generating Metal IR for:", src_path)
    print("  Command:", command)
    system(command)
    print("  Generated Metal IR file:", ir_path)

ir_file_paths = " ".join(ir_file_path(name) for name in kernel_names)
link_command = "xcrun -sdk macosx metallib {} -o {}kernels.metallib".format(ir_file_paths, binary_directory)
print("Generating Metal library")
print("  Command:", link_command)
system(link_command)
print("  Generated Metal library: {}kernels.metallib".format(binary_directory))
