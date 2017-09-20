#!/bin/sh

input=$1
output=$2

value=`awk '{if($1=="DEFS"){print $3,$4,$5}}' ${input}`
echo "#pragma once" > ${output}
for x in ${value};do
    sym=${x:2:${#x}}
    no_eq=`echo ${sym} | sed 's/=/ /g'`
    name=`echo ${no_eq} | awk '{print $1}'`
    value=`echo ${no_eq} | awk '{print $2}'`
    echo "#ifndef ${name}">> ${output}
    echo "#define ${name} ${value}" >> ${output}
    echo "#endif">>${output}
done
echo "#include<ctf/sub_dir/ctf.hpp>" >> ${output}
