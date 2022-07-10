OMP_NUM_THREADS=2 MKL_NUM_THREADS=2  python3 ../scripts/start_training.py \
    --arch custom_resnet18 \
    --loss soft-labels \
    --data inaturalist19-224 \
    --beta 30 \
    --workers 16 \
    --val_freq 1 \
    --data-paths-config ../data_paths.yml \
    --output softlabels_inaturalist19_beta30_customresnet/ \
    --num_training_steps 200000

