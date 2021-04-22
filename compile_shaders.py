import os
import re
import sys
import glob
import json
import gzip

from collections import defaultdict

os.chdir('/'.join(__file__.replace('\\', '/').split('/')[:-1]))
print(os.getcwd())
cmd_remove = ''
null_out = ''

if sys.platform.find('win32') != -1:
    cmd_remove = 'del'
    null_out = ' >>nul 2>nul'
    dir = os.path.join('\\'.join(__file__.split('\\')[:-1]), 'shaders')
    print(dir)

elif sys.platform.find('linux') != -1:
    cmd_remove = 'rm'
    null_out = ' > /dev/null 2>&1'
    dir = os.path.join('/'.join(__file__.split('/')[:-1]), 'shaders')
    print(dir)

headfile = open('./vknn/spv_shader.h', 'w+')
cpp_file = open('./vknn/spv_shader.cpp', 'w+')

print()

lst = list()
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".comp"):
            lst.append(os.path.join(root, file))

outfile_str = ['#include <cstdlib>\n\n']
bin_code = list()
bin_dict = {}
forced = True

if(os.path.exists('compile.json.gz')):
    with gzip.GzipFile('compile.json.gz', 'r') as fin:
        bin_dict = json.loads(fin.read().decode('utf-8'))

print(len(bin_dict))
dir_change = len(bin_dict) != len(lst)

for i in range(0, len(lst)):
    path = lst[i]
    prefix = os.path.splitext(os.path.split(path)[-1])[0]
    array_name = prefix + '_spv'
    spv_txt_file = prefix + '.spv'
    if(prefix not in bin_dict.keys()):
        bin_dict[prefix] = {'time':0, 'bin':[], 'header':''}
    modified = bin_dict[prefix]['time'] != os.path.getmtime(path)
    print(prefix, modified)
    if(modified or dir_change or forced):
        bin_dict[prefix]['time'] = os.path.getmtime(path)
        bin_file = prefix + '.tmp'
        cmd = 'glslangValidator --target-env spirv1.3 -V ' + path + ' -S comp -o ' + bin_file
        if os.system(cmd) != 0:
            continue

        cmd = 'glslangValidator --target-env spirv1.3 -V ' + path + ' -S comp -o ' + spv_txt_file + ' -x' + null_out
        os.system(cmd)
        bin_dict[prefix]['bin'] = []
        infile = open(spv_txt_file, 'r')
        bin_dict[prefix]['bin'].append("\n")
        size = os.path.getsize(bin_file)
        fmt = 'extern const unsigned int %s[%d] = {\n' % (array_name, size / 4)
        bin_dict[prefix]['bin'].append(fmt)
        for eachLine in infile:
            if(re.match(r'^.*\/\/', eachLine)):
                continue
            newline = "\t" + eachLine.replace('\t','')
            bin_dict[prefix]['bin'].append(newline)

        infile.close()
        bin_dict[prefix]['bin'].append("};\n")
        bin_dict[prefix]['header'] = "extern const unsigned int %s[%d];\n" % (array_name, size / 4)
        os.system(cmd_remove + ' ' + bin_file)
        os.system(cmd_remove + ' ' + spv_txt_file)

    for line in bin_dict[prefix]['bin']:
        bin_code.append(line)
    outfile_str.append(bin_dict[prefix]['header'])

with gzip.GzipFile('compile.json.gz', 'w') as fout:
    fout.write(json.dumps(bin_dict).encode('utf-8'))

headfile.writelines(outfile_str + ["\n"])
cpp_file.writelines(['#include<cstdlib>\n#include "spv_shader.h"\n'] + bin_code)

for root, dirs, files in os.walk(dir):
    for currentFile in files:
        exts = ('.spv', '.tmp')
        if currentFile.lower().endswith(exts):
            os.remove(os.path.join(root, currentFile))

cpp_file.close()
headfile.close()