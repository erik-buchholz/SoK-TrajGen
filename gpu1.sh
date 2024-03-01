#!/usr/bin/env bash

#python3 -m stg.run.train_cnn_gan -g 1 --dataset fs --wgan --gp --lp --lr_d 3e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000
#python3 -m stg.run.train_cnn_gan -g 1 --dataset geolife --wgan --gp --lp --lr_d 3e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000
#python3 -m stg.run.train_cnn_gan -g 1 --dataset fs --lr_d 1e-4 --lr_g 1e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000
#python3 -m stg.run.train_cnn_gan -g 1 --dataset geolife --lr_d 1e-4 --lr_g 1e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000

# Run LSTM  GAN on GPU 1
python3 -m stg.eval.lstm_gan -g 1 -d fs -b 2000
python3 -m stg.eval.lstm_gan -g 1 -d geolife -b 2000
python3 -m stg.eval.lstm_gan -g 1 -d geolife -b 2000 --spatial