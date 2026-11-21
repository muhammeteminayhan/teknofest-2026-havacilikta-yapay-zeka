#!/bin/bash
P=/c/Users/merto/Desktop/havacilikta-yz-project
cd $P
for AD in video1 final_o1 final_o3 final_o4 final_o2 final_test video2; do
  echo "=== $AD $(date +%H:%M:%S)"
  $P/.venv_colmap/Scripts/python.exe -u $P/gorev2/sozde_gt.py $AD --dongu > $P/gorev2/outputs/final/colmap/$AD/dongu.log 2>&1
  grep "^\[" $P/gorev2/outputs/final/colmap/$AD/dongu.log | tail -8
done
echo "=== BITTI $(date +%H:%M:%S)"
