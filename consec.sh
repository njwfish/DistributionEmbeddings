for enc in conv_gnn kernel_mean; do
  python main.py experiment=fmnist_sys_diff encoder=$enc training.max_time=900
done
python main.py experiment=fmnist_sys_wormhole encoder=wormhole_encoder training.max_time=900

for enc in conv_gnn kernel_mean; do
  python main.py experiment=mnist_sys_diff encoder=$enc training.max_time=900
done
python main.py experiment=mnist_sys_wormhole encoder=wormhole_encoder training.max_time=900

for enc in resnet kernel_mean; do
  python main.py experiment=gmm_sys_diff encoder=$enc training.max_time=600
done
python main.py experiment=gmm_sys_wormhole encoder=wormhole_encoder training.max_time=600