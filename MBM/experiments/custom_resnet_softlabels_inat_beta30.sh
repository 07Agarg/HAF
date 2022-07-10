OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 ../scripts/start_training.py \
 --arch custom_ resnet18 --loss soft-labels --lr 1e-5 --data inaturalist19-224 --beta 30 --workers 16 --data-paths-config data_paths.yml --num_training_steps 200000
