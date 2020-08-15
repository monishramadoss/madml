import os
import re
import sys
import glob

os.chdir('/'.join(__file__.replace('\\', '/').split('/')[:-1]))
print(os.getcwd())
cmd_remove = ''
null_out = ''


if sys.platform.find('win32') != -1:
    cmd_remove = 'del'
    null_out = ' >>nul 2>nul'
    dir = os.path.join('\\'.join(__file__.split('\\')[:-1]), 'halaml', 'shaders')
    print(dir)

elif sys.platform.find('linux') != -1:
    cmd_remove = 'rm'
    null_out = ' > /dev/null 2>&1'
    dir = os.path.join('/'.join(__file__.split('/')[:-1]), 'halaml', 'shaders')
    print(dir)

headfile = open('./halaml/spv_shader.h', 'w+')
cpp_file = open('./halaml/spv_shader.cpp', 'w+')
lst = list()
for root, dirs, files in os.walk("./"):
    for file in files:
        if file.endswith(".comp"):
             lst.append(os.path.join(root, file))

outfile_str = ['#include "pch.h"\n#include <cstdlib>\n\nnamespace kernel { \n\tnamespace shaders {\n']
bin_code = list()

for i in range(0, len(lst)):
    prefix = os.path.splitext(os.path.split(lst[i])[-1])[0]
    path = lst[i]

    bin_file = prefix + '.tmp'
    cmd = ' glslangValidator -V ' + path + ' -S comp -o ' + bin_file
    if os.system(cmd) != 0:
        continue

    size = os.path.getsize(bin_file)
    spv_txt_file = prefix + '.spv'
    cmd = 'glslangValidator -V ' + path + ' -S comp -o ' + spv_txt_file + ' -x' + null_out 
    os.system(cmd)
    infile_name = spv_txt_file

    array_name = prefix + '_spv'
    infile = open(infile_name, 'r')
    bin_code.append('\nnamespace kernel { \n\tnamespace shaders {\n')
    fmt = '\t\textern const unsigned int %s[%d] = {\n' % (array_name, size/4)
    bin_code.append(fmt)
    for eachLine in infile:
        if(re.match(r'^.*\/\/', eachLine)):
            continue
        newline = '\t\t\t' + eachLine.replace('\t','')
        bin_code.append(newline)
    infile.close()
    bin_code.append("\t\t};\n\t}\n} //namespace kernel, shaders\n\n")
    outfile_str.append('\t\textern const unsigned int %s[%d];\n' % (array_name, size/4))
    os.system(cmd_remove + ' ' + bin_file)
    os.system(cmd_remove + ' ' + spv_txt_file)

headfile.writelines(outfile_str  + ["\t}\n}"])
cpp_file.writelines(['#include<cstdlib>\n#include "spv_shader.h"'] + bin_code);

cpp_file.close()
headfile.close()
