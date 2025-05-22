for i in {5..5}; do

    # python main.py --multirun experiment=mvn_sys_diff encoder=mean,kernel_mean,gnn,median_gnn,resnet,tx experiment.name="mvn_sys_diff_$i" training.max_time=300 dataset.prior_cov_scale=0.1

    python main.py --multirun experiment=mvn_sys_sinkhorn encoder=mean,kernel_mean,gnn,median_gnn,resnet,tx experiment.name="mvn_sys_sinkhorn_$i" training.max_time=300 dataset.prior_cov_scale=0.1

    python main.py --multirun experiment=mvn_sys_sw encoder=mean,kernel_mean,gnn,median_gnn,resnet,tx experiment.name="mvn_sys_sw_$i" training.max_time=300 dataset.prior_cov_scale=0.1

    python main.py --multirun experiment=mvn_sys_vae encoder=mean,kernel_mean,gnn,median_gnn,resnet,tx experiment.name="mvn_sys_vae_$i" training.max_time=300 dataset.prior_cov_scale=0.1

    python main.py --multirun experiment=mvn_sys_wormhole encoder=mean,kernel_mean,gnn,median_gnn,resnet,tx experiment.name="mvn_sys_wormhole_$i" training.max_time=300 dataset.prior_cov_scale=0.1

done
