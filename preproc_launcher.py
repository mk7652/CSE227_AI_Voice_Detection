import os
import shutil

charsiu_dir = "charsiu"

if not os.path.isdir(charsiu_dir):
    os.system("git clone https://github.com/lingjzhu/charsiu")

pre_proc_script = "preproc.py"

if not os.path.isfile(charsiu_dir + '\\' + pre_proc_script):
    shutil.move(pre_proc_script, charsiu_dir)

os.chdir('charsiu')

os.system("python preproc.py")