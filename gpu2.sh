# RAoPT vs LSTM-TrajGAN
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset fs -g 2 -p 3 --batch_size 128 --early_stop 250  --epochs 1500
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset geolife -g 2 -p 3 --batch_size 512 --epochs 1500 --latlon_only
python3 -m stg.eval.raopt_vs_lstm --runs 5 --dataset geolife -g 2 -p 3 --batch_size 512 --epochs 1500