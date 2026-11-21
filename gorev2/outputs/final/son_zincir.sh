#!/bin/bash
P=/c/Users/merto/Desktop/havacilikta-yz-project; F=$P/gorev2/outputs/final; cd $P
bekle() { until [ -f $P/gorev2/outputs/nokta_raft_$1_sp8.npz ] && [ -f $F/colmap/$1/sozde_gt.npz ] && [ -f $F/colmap/$1/orbslam3_poz.csv ]; do sleep 30; done; }
# O1/O3/O4 kosusu bitsin (json yazimi cakismasin)
while pgrep -f "final_degerlendir.py --oturum final_o1" >/dev/null 2>&1 || ps -W | grep -q "[f]inal_degerlendir"; do sleep 30; done
bekle final_o2; echo "O2 $(date +%H:%M)"; $P/.venv/Scripts/python.exe -u $P/gorev2/final_degerlendir.py --oturum final_o2 --n-perm 3 > $F/degerlendir_o2.log 2>&1
bekle final_test; echo "TEST $(date +%H:%M)"; $P/.venv/Scripts/python.exe -u $P/gorev2/final_degerlendir.py --oturum final_test --n-perm 3 > $F/degerlendir_test.log 2>&1
$P/.venv/Scripts/python.exe -u $P/gorev2/final_rapor.py > $F/final_rapor.log 2>&1
echo "BITTI $(date +%H:%M)"
