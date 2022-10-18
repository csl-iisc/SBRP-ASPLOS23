mkdir -p results
rm -f results/*
make > ./compile_log.txt

echo "Executing volatile..."
./build/imkv_gpu > ./results/volatile.txt
echo "Executing GPM-far..."
./build/imkv_real_cpu > ./results/gpm_far.txt
echo "Executing GPM-near..."
./build/imkv_emul_gpu > ./results/gpm_near.txt
echo "Executing Coarse-fs..."
./build/imkv_fs_gpu > ./results/coarse_fs.txt
echo "Executing Coarse-mm..."
./build/imkv_mm_gpu > ./results/coarse_mm.txt
echo "Executing Coarse-mm-tx..."
./build/imkv_mm_tx_gpu > ./results/coarse_mm_tx.txt
#echo "Executing Coarse-tx..."
#%./imkv_tx_gpu > ./results/coarse_tx.txt
echo "Done!"
