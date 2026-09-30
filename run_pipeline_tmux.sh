#!/usr/bin/env bash

SESSION="tcp_rl"
PROJECT_DIR="$HOME/ns-allinone-3.40/ns-3.40/tcp_rl_project"
NS3_DIR="$HOME/ns-allinone-3.40/ns-3.40"
VENV_DIR="$HOME/ns-allinone-3.40/ns-3.40/venv"
PORT=7144
DURATION=6500

# Terminate existing session or orphaned background processes
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -f "rl-tcp" || true
pkill -f "python.*train.py" || true
sleep 1

# Purge previous execution artifacts
cd "$PROJECT_DIR"
rm -f *.zip *.csv *.png *.pkl

# Create a new detached tmux session with the left pane for ns-3
tmux new-session -d -s "$SESSION" -n "pipeline"
tmux send-keys -t "$SESSION:0.0" "cd $NS3_DIR && ./ns3 run 'rl-tcp --openGymPort=${PORT} --duration=${DURATION}'" C-m

# Split horizontally: right side will run Python training
tmux split-window -h -t "$SESSION:0"
tmux send-keys -t "$SESSION:0.1" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate" C-m

# Wait for ZeroMQ port binding on 7144 before launching train.py
tmux send-keys -t "$SESSION:0.1" "until nc -z localhost ${PORT} 2>/dev/null; do sleep 0.5; done" C-m
tmux send-keys -t "$SESSION:0.1" "python train.py" C-m

# Split bottom-right pane to automatically plot and display results when traces appear
tmux split-window -v -t "$SESSION:0.1"
tmux send-keys -t "$SESSION:0.2" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate" C-m
tmux send-keys -t "$SESSION:0.2" "while [ ! -f rl_tcp_traces.csv ]; do sleep 2; done; sleep 2; python plot_results.py && xdg-open rl_tcp_zoomed_performance.png" C-m

# Attach directly to the visual session
tmux attach-session -t "$SESSION"
