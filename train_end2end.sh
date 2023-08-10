export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd src

torchrun --nproc_per_node=8 main_end2end_v2s.py
# torchrun --nproc_per_node=8 main_end2end_convnext.py
