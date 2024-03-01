NAME=Project2
CONDA_ENV=stg
PROJECT_DIR=$(dirname "$(realpath "$0")")
tmux new-session -s ${NAME} -d
tmux renamew 'Monitoring'
tmux split-window -v -t ${NAME}:0. -c ~ -d
tmux send-keys -t ${NAME}:0.0 "gpustat -i" C-m
tmux send-keys -t ${NAME}:0.1 "htop" C-m
tmux new-window -t ${NAME} -n 'Jupyter/TensorBoard RAoPT' -c ${PROJECT_DIR} -d
tmux split-window -v -t ${NAME}:1. -c ${PROJECT_DIR} -d
tmux send-keys -t ${NAME}:1.0 "conda activate ${CONDA_ENV};jupyter notebook --port=9999 --no-browser --NotebookApp.token=''" C-m
tmux send-keys -t ${NAME}:1.1 "conda activate ${CONDA_ENV};tensorboard --logdir ${PROJECT_DIR}runs" C-m
tmux new-window -t ${NAME} -n 'GPU #0' -c ${PROJECT_DIR} -d
tmux send-keys -t ${NAME}:2 "conda activate ${CONDA_ENV}" C-m
tmux new-window -t ${NAME} -n 'GPU #1' -c ${PROJECT_DIR} -d
tmux send-keys -t ${NAME}:3 "conda activate ${CONDA_ENV}" C-m
tmux new-window -t ${NAME} -n 'GPU #2' -c ${PROJECT_DIR} -d
tmux send-keys -t ${NAME}:4 "conda activate ${CONDA_ENV}" C-m
tmux new-window -t ${NAME} -n 'GPU #3' -c ${PROJECT_DIR} -d
tmux send-keys -t ${NAME}:5 "conda activate ${CONDA_ENV}" C-m