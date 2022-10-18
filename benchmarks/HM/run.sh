mkdir -p results
rm -f results/*
make > ./compile_log.txt

echo "Executing volatile..."
./build/hashmap_gpu > ./results/volatile.txt
echo "Executing GPM-far..."
./build/hashmap_real_cpu > ./results/gpm_far.txt
echo "Executing GPM-far, no HCL..."
./build/hashmap_real_no-hcl_cpu > ./results/gpm_far_no-hcl.txt
echo "Executing GPM-near..."
./build/hashmap_emul_gpu > ./results/gpm_near.txt
#echo "Executing Coarse-fs..."
#./build/hashmap_fs_gpu > ./results/coarse_fs.txt
echo "Executing Coarse-mm..."
./build/hashmap_mm_gpu > ./results/coarse_mm.txt
#echo "Executing Coarse-mm-tx..."
#./build/hashmap_mm_tx_gpu > ./results/coarse_mm_tx.txt
echo "Done!"
