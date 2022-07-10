OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 main.py --start testing --arch wide_resnet --loss soft-labels --optimizer adam_amsgrad --data cifar-100 --workers 16 --output out/cifar-100/soft-labels
