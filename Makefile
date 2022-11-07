SHELL := /bin/bash  
CUDA_INSTALL_PATH ?= /usr/local/cuda-11.4

figure6_models 	  = epoch_far sbrp_far epoch_near sbrp_near
figure9_models    = epoch_far epoch_far_eadr sbrp_far_eadr
figure10_a_models = sbrp_near_L1_125 sbrp_near_L1_25 sbrp_near_L1_50 sbrp_near epoch_near
figure10_b_models = epoch_near_bw_half sbrp_near_bw_half sbrp_near epoch_near_bw_double sbrp_near_bw_double epoch_near
figure10_c_models = sbrp_near_window_2 sbrp_near_window_4 sbrp_near sbrp_near_window_8 sbrp_near_window_10 sbrp_near_window_12 epoch_near
figure11_models   = epoch_near sbrp_near 
gpm_model 		  = gpm


models ?= $(figure6_models)
results ?= results

.SILENT: run_output_all
.PHONY: run_output_all

default: 
	make run_output_all
	
clean_run:
	#rm -rf results/;
	mkdir -p $(results)/;
	
output:
ifdef benchmark
ifndef model
	echo $(benchmark)
	for j in $(models); do \
		time=$$(tac $(results)//$${j}_$(benchmark).out | grep -m1 "gpu_tot_sim_cycle =" | grep -m1 -oE '[0-9]+'); \
		w_misses=$$(tac $(results)//$${j}_$(benchmark).out | grep -m1 "L1D_total_pm_write_cache_misses =" | grep -m2 -oE '[0-9]+[0-9]+'); \
		r_misses=$$(tac $(results)//$${j}_$(benchmark).out | grep -m1 "L1D_total_pm_read_cache_misses =" | grep -m2 -oE '[0-9]+[0-9]+'); \
		bw_utils=$$(tac $(results)//$${j}_$(benchmark).out | grep -m12 -oE "bw_util=[0-9]+\.[0-9]+" | grep -m12 -oE '[0-9]+\.[0-9]+'); \
		avg_util=0.0; \
		for x in $${bw_utils}; do \
			avg_util=$$(echo "$$x+$$avg_util" | bc); \
		done; \
		avg_util=$$(echo "scale=5;$$avg_util / 12.0" | bc); \
		printf '%s:\t%d\tcycles\tPM BW Util:\t%f\tWMisses:\t%d\tRMisses:\t%d\n' $${j} $${time} $${avg_util} $${w_misses} $${r_misses}; \
	done
else		
	time=$(tac $(results)//$(model)_$(benchmark).out | grep -m1 "gpu_tot_sim_cycle =" | grep -m1 -oE '[0-9]+'); \
	accuracy=$(tac $(results)//$(model)_$(benchmark).out | grep -m1 "L1D_total_cache_miss_rate =" | grep -m1 -oE '[0-9]+\.[0-9]+'); \
	bw_utils=$$(tac $(results)//$(model)_$(benchmark).out | grep -m12 "bw_util =" | grep -m1 -oE '[0-9]+\.[0-9]+'); \
	avg_util=0; \
	for x in $${bw_utils}; do \
		avg_util+=$$x; \
	done \
	avg_util/=12; \
	printf '%s:\t%d\tcycles\tPM BW Util:\t%f\n' $(model) $${time} $${avg_util};
endif
endif

run_benchmark:
ifndef folder
	echo "Missing 'folder' in make run_benchmark"
else
ifndef benchmark
	echo "Missing 'benchmark' name in make run_benchmark"
else
ifndef exec_command
	echo "Missing 'exec_command' in make run_benchmark"
else
	$(MAKE) -s clean_run results=$(results)
	cd ./benchmarks/$(folder)/; \
	$(MAKE)
	
	cd ./simulator;\
	export CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH);\
	source setup_environment;\
	$(MAKE) -s; \
	cd ../models; \
	for j in $(models); do \
		cd $$j; \
		echo "Running $${j} $(benchmark)";\
		../../benchmarks/$(folder)/$(exec_command) > ../../$(results)/$${j}_$(benchmark).out & \
		cd ..;\
	done; \
	wait;
	#$(MAKE) output benchmark=$(benchmark)
endif
endif
endif

run_KVS: 
	$(MAKE) -s run_benchmark folder=KVS benchmark=KVS exec_command="./build/KVS_gpu" models="$(models)" results=$(results)

run_KVS_gpm: 
	$(MAKE) -s run_benchmark folder=KVS benchmark=KVS exec_command="./build/KVS_gpm" models="$(gpm_model)" results=$(results)

run_KVS_rec: 
	$(MAKE) -s run_benchmark folder=KVS benchmark=KVS_rec exec_command="./build/KVS_rec" models="$(figure11_models)" results=$(results)

run_SRAD: 
	$(MAKE) -s run_benchmark folder=SRAD benchmark=SRAD exec_command="./build/SRAD_gpu 512 512 0 31 0 31 0.5 1" models="$(models)" results=$(results)

run_SRAD_rec: 
	$(MAKE) -s run_benchmark folder=SRAD benchmark=SRAD_rec exec_command="./build/SRAD_rec 512 512 0 31 0 31 0.5 1" models="$(figure11_models)" results=$(results)

run_SRAD_gpm: 
	$(MAKE) -s run_benchmark folder=SRAD benchmark=SRAD exec_command="./build/SRAD_gpm 512 512 0 31 0 31 0.5 1" models="$(gpm_model)" results=figure6_results

run_HM: 
	$(MAKE) -s run_benchmark folder=HM benchmark=HM exec_command="./build/HM_gpu" models="$(models)" results=$(results)

run_HM_rec:
	$(MAKE) -s run_benchmark folder=HM benchmark=HM_rec exec_command="./build/HM_rec" models="$(figure11_models)" results=$(results)

run_HM_gpm: 
	$(MAKE) -s run_benchmark folder=HM benchmark=HM exec_command="./build/HM_gpm" models="$(gpm_model)" results=$(results)

run_Red:
	$(MAKE) -s run_benchmark folder=Red benchmark=Red exec_command="./build/Red_gpu" models="$(models)" results=$(results)

run_Red_rec:
	$(MAKE) -s run_benchmark folder=Red benchmark=Red_rec exec_command="./build/Red_rec" models="$(figure11_models)" results=$(results)

run_Red_gpm:
	$(MAKE) -s run_benchmark folder=Red benchmark=Red exec_command="./build/Red_gpm" models="$(gpm_model)" results=$(results)

run_Scan:
	$(MAKE) -s run_benchmark folder=Scan benchmark=Scan exec_command="./build/Scan_gpu" models="$(models)" results=$(results)

run_Scan_rec:
	$(MAKE) -s run_benchmark folder=Scan benchmark=Scan_rec exec_command="./build/Scan_rec" models="$(figure11_models)" results=$(results)

run_Scan_gpm:
	$(MAKE) -s run_benchmark folder=Scan benchmark=Scan exec_command="./build/Scan_gpm" models="$(gpm_model)" results=$(results)

run_MQ:
	$(MAKE) -s run_benchmark folder=MQ benchmark=MQ exec_command="./build/MQ_gpu" models="$(models)" results=$(results)

run_MQ_rec:
	$(MAKE) -s run_benchmark folder=MQ benchmark=MQ_rec exec_command="./build/MQ_rec" models="$(figure11_models)" results=$(results)

run_MQ_gpm:
	$(MAKE) -s run_benchmark folder=MQ benchmark=MQ exec_command="./build/MQ_gpm" models="$(gpm_model)" results=$(results)


#run_benchmark:
#	mkdir -p $(results) 
#	$(make) run_KVS  models="$(models)" results=$()
#	$(make) run_HM	 models="$(models)" results=$()
#	$(make) run_SRAD models="$(models)" results=$()
#	$(make) run_Red  models="$(models)" results=$()
#	$(make) run_MQ   models="$(models)" results=$()
#	$(make) run_Scan models="$(models)" results=$()

#run_figure6: 
#	$(MAKE) run_benchmark models=$(figure6_models) results=figure6_results
#	$(MAKE) run_KVS_gpm models="gpm" results=figure6_results
#	$(MAKE) run_HM_gpm models="gpm" results=figure6_results
#	$(MAKE) run_SRAD_gpm models="gpm" results=figure6_results
#	$(MAKE) run_Red_gpm models="gpm" results=figure6_results
#	$(MAKE) run_MQ_gpm models="gpm" results=figure6_results
#	$(MAKE) run_Scan_gpm models="gpm" results=figure6_results

run_output_all: 
	$(MAKE) run_all
	$(MAKE) output_all 

run_figure6: 
	mkdir -p figure6_results
	$(MAKE) run_KVS models="$(figure6_models)" results=figure6_results
	$(MAKE) run_KVS_gpm models="gpm" results=figure6_results
	$(MAKE) run_HM models="$(figure6_models)" results=figure6_results
	$(MAKE) run_HM_gpm models="gpm" results=figure6_results
	$(MAKE) run_SRAD models="$(figure6_models)" results=figure6_results
	$(MAKE) run_SRAD_gpm models="gpm" results=figure6_results
	$(MAKE) run_Red models="$(figure6_models)" results=figure6_results
	$(MAKE) run_Red_gpm models="gpm" results=figure6_results
	$(MAKE) run_MQ models="$(figure6_models)" results=figure6_results
	$(MAKE) run_MQ_gpm models="gpm" results=figure6_results
	$(MAKE) run_Scan models="$(figure6_models)" results=figure6_results
	$(MAKE) run_Scan_gpm models="gpm" results=figure6_results

run_figure9:
	mkdir -p figure9_results
	$(MAKE) run_KVS models="$(figure9_models)" results=figure9_results
	$(MAKE) run_HM models="$(figure9_models)" results=figure9_results
	$(MAKE) run_SRAD models="$(figure9_models)" results=figure9_results
	$(MAKE) run_Red models="$(figure9_models)" results=figure9_results
	$(MAKE) run_MQ models="$(figure9_models)" results=figure9_results
	$(MAKE) run_Scan models="$(figure9_models)" results=figure9_results

run_figure10_a:
	mkdir -p figure10_a_results
	$(MAKE) run_KVS models="$(figure10_a_models)" results=figure10_a_results
	$(MAKE) run_HM models="$(figure10_a_models)" results=figure10_a_results
	$(MAKE) run_SRAD models="$(figure10_a_models)" results=figure10_a_results
	$(MAKE) run_Red models="$(figure10_a_models)" results=figure10_a_results
	$(MAKE) run_MQ models="$(figure10_a_models)" results=figure10_a_results
	$(MAKE) run_Scan models="$(figure10_a_models)" results=figure10_a_results

run_figure10_b:
	mkdir -p figure10_b_results
	$(MAKE) run_KVS models="$(figure10_b_models)" results=figure10_b_results
	$(MAKE) run_HM models="$(figure10_b_models)" results=figure10_b_results
	$(MAKE) run_SRAD models="$(figure10_b_models)" results=figure10_b_results
	$(MAKE) run_Red models="$(figure10_b_models)" results=figure10_b_results
	$(MAKE) run_MQ models="$(figure10_b_models)" results=figure10_b_results
	$(MAKE) run_Scan models="$(figure10_b_models)" results=figure10_b_results

run_figure10_c:
	mkdir -p figure10_c_results
	$(MAKE) run_KVS models="$(figure10_c_models)" results=figure10_c_results
	$(MAKE) run_HM models="$(figure10_c_models)" results=figure10_c_results
	$(MAKE) run_SRAD models="$(figure10_c_models)" results=figure10_c_results
	$(MAKE) run_Red models="$(figure10_c_models)" results=figure10_c_results
	$(MAKE) run_MQ models="$(figure10_c_models)" results=figure10_c_results
	$(MAKE) run_Scan models="$(figure10_c_models)" results=figure10_c_results

run_figure11:
	mkdir -p figure11_results
	$(MAKE) run_KVS_rec models="$(figure11_models)" results=figure11_results
	$(MAKE) run_HM_rec models="$(figure11_models)" results=figure11_results
	$(MAKE) run_SRAD_rec models="$(figure11_models)" results=figure11_results
	$(MAKE) run_Red_rec models="$(figure11_models)" results=figure11_results
	$(MAKE) run_MQ_rec models="$(figure11_models)" results=figure11_results
	$(MAKE) run_Scan_rec models="$(figure11_models)" results=figure11_results

run_all: 
	make run_figure6
	make run_figure9
	make run_figure10_a
	make run_figure10_b
	make run_figure10_c
	make run_figure11

output_figure6: 
	mkdir -p outputs/ && cd scripts/ && python3 figure6_plot.py && cd ../ 

output_figure8: 
	mkdir -p outputs/ && cd scripts/ && 	python3 figure8_plot.py && cd ../ 

output_figure9: 
	mkdir -p outputs/ && cd scripts/ && 	python3 figure9_plot.py && 	cd ../

output_figure10_a: 
	mkdir -p outputs/ && cd scripts/ && 	python3 figure10a_plot.py && cd ../ 

output_figure10_b: 
	mkdir -p outputs/ && cd scripts/ && 	python3 figure10b_plot.py && cd ../ 

output_figure10_c: 
	mkdir -p outputs/ && cd scripts/ &&  python3 figure10c_plot.py && cd ../ 

output_figure11: 
	mkdir -p outputs/ && cd scripts/ && 	python3 figure11_plot.py && cd ../ 

output_all: 
	mkdir -p outputs/; 
	make output_figure6
	make output_figure8
	make output_figure9
	make output_figure10_a
	make output_figure10_b
	make output_figure10_c
	make output_figure11

clean:
	rm -f $(outputs)
	rm -f */_app*
	rm -f */_cuo*
	rm -f */*.ptx*


