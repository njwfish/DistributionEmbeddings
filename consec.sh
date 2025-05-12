for enc in wormhole_encoder conv_gnn kernel_mean mean; do
  python main.py experiment=fmnist_sys_diff encoder=$enc
done

for enc in wormhole_encoder conv_gnn kernel_mean mean; do
  python main.py experiment=fmnist_sys_wormhole encoder=$enc
done

for enc in wormhole_encoder conv_gnn kernel_mean mean; do
  python main.py experiment=fmnist_sys_vae encoder=$enc
done