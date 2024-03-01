# Eval 1 - FS NYC
#python3 -m stg.run.train_cnn_gan -g 0 --dataset mnist_sequential --wgan --gp --lp --lr_d 3e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000
#python3 -m stg.run.train_cnn_gan -g 0 --dataset mnist_sequential --lr_d 1e-4 --lr_g 1e-4 --n_critic 1 -e 200 --save_freq 100 -p 1000

# LSTM-Convergence Eval
# Case 1: FS NYC with 2,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset fs -b 2000
# Case 2: FS NYC with 20,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset fs -b 20000
# Case 3: Geolife with 2,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 2000
# Case 4: Geolife with 20,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 20000
# Case 5: Geolife Spatial with 2,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 2000 --spatial
# Case 6: Geolife Spatial with 20,000 batches
python3 -m stg.eval.lstm_convergence -g 0 --dataset geolife -b 20000 --spatial
