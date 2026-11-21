#!/bin/bash
for AD in final_o4 final_o1; do for R in r2 r3; do
  "/c/Program Files/Docker/Docker/resources/bin/docker.exe" run --rm -v 'C:\Users\merto\Desktop\havacilikta-yz-project\gorev2\outputs\final:/veri' orbslam3:yerel mono_nedensel /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt /veri/colmap/$AD/orbslam3.yaml /veri/colmap/$AD/images /veri/colmap/$AD/orbslam3_poz_$R.csv 7.5 > /c/Users/merto/Desktop/havacilikta-yz-project/gorev2/outputs/final/colmap/$AD/orbslam3_kos_$R.log 2>&1
  echo "$AD $R $(date +%H:%M)"
done; done
