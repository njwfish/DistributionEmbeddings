for enc in conv_gnn kernel_mean; do
  python main.py experiment=cifar_sys_diff encoder=$enc training.max_time=1200
done
python main.py experiment=cifar_sys_wormhole encoder=wormhole_encoder training.max_time=1200