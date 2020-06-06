# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id 18yvyft4 --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id 12s5inal --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id 34034jpo --use-wandb

# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/train/feature_regression_training.py \
#     --foggy_beta beta_0.005 --total-epoch 12 --feature-regression-criteria MSELoss --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/train/feature_regression_training.py \
#     --foggy_beta beta_0.01 --total-epoch 12 --feature-regression-criteria MSELoss --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/train/feature_regression_training.py \
#     --foggy_beta beta_0.02 --total-epoch 12 --feature-regression-criteria MSELoss --use-wandb

# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id x8pm1nod --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id dpiqz1se --use-wandb
# env /opt/conda/envs/py37torch15/bin/python /home/user/research/refinenet-pytorch/eval/feature_regression_evaluating.py \
#     --wandb_id tlr7wxnf --use-wandb

env /opt/conda/envs/py37torch15/bin/python \
    /home/user/research/refinenet-pytorch/train/feature_regression_training.py \
        --clear-foggy-beta beta_0.005 \
        --update-period -1 --feature-regression-criteria L1Loss --feature-regression-target-weights 1 1 --feature_layer refinenets \
        --data-aug-hflip --data-aug-hflip-p 0.5 \
        --data-aug-crop --data-aug-crop-size 600 600 --data-aug-crop-scale 0.7 1.3 --data-aug-crop-ratio 1 1 \
        --input-scale-factor 1. --total-epoch 12 --batch-size 1 --valid-batch-size 1 \
        --freeze-batch-norm --use-wandb