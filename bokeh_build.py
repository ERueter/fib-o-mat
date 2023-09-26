import sys
import shutil
import subprocess as sp
import pathlib

temp_dir = pathlib.Path(sys.argv[1])
out_dir = pathlib.Path(sys.argv[2])
out_file = pathlib.Path(sys.argv[3])
input_files = sys.argv[4:]

temp_dir /= out_file.name[: -len("".join(out_file.suffixes))]


print("temp_dir =", temp_dir)
print("out_dir =", out_dir)
print("out_file =", out_file)
print("input_files =", input_files)


temp_dir.mkdir(exist_ok=True, parents=True)

for src in input_files:
    shutil.copy2(src, temp_dir)

sp.check_call(["bokeh", "build", "--rebuild", temp_dir])

shutil.copy2(temp_dir / "dist" / out_file, out_dir / out_file)
