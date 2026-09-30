# 1. Force kill the entire stuck tmux server and ns-3 simulation immediately
tmux kill-server
pkill -9 -f "rl-tcp"
pkill -9 -f "python.*train.py"
pkill -9 -f "nc -z"

# 2. Overwrite run_pipeline_tmux.sh with a clean, bulletproof launcher
cat << 'EOF' > ~/ns-allinone-3.40/ns-3.40/tcp_rl_project/run_pipeline_tmux.sh
#!/usr/bin/env bash

SESSION="tcp_rl"
PROJECT_DIR="$HOME/ns-allinone-3.40/ns-3.40/tcp_rl_project"
NS3_DIR="$HOME/ns-allinone-3.40/ns-3.40"
VENV_DIR="$HOME/ns-allinone-3.40/ns-3.40/venv"
PORT=7144
DURATION=6500

# Cleanup
tmux kill-session -t "$SESSION" 2>/dev/null || true
pkill -9 -f "rl-tcp" || true
pkill -9 -f "python.*train.py" || true
sleep 1

# Purge previous run traces
cd "$PROJECT_DIR"
rm -f *.zip *.csv *.png *.pkl

# Start tmux with --norc to suppress fastfetch/system info clutter
tmux new-session -d -s "$SESSION" -n "pipeline" "bash --norc"
tmux send-keys -t "$SESSION:0.0" "clear && cd $NS3_DIR && ./ns3 run 'rl-tcp --openGymPort=${PORT} --duration=${DURATION}'" C-m

# Right Pane: PPO training
tmux split-window -h -t "$SESSION:0" "bash --norc"
tmux send-keys -t "$SESSION:0.1" "clear && cd $PROJECT_DIR && source $VENV_DIR/bin/activate && sleep 3 && python train.py" C-m

# Bottom Right Pane: Auto-plotter watcher
tmux split-window -v -t "$SESSION:0.1" "bash --norc"
tmux send-keys -t "$SESSION:0.2" "clear && cd $PROJECT_DIR && source $VENV_DIR/bin/activate && while [ ! -f rl_tcp_traces.csv ]; do sleep 2; done; sleep 2; python plot_results.py && xdg-open rl_tcp_zoomed_performance.png" C-m

tmux attach-session -t "$SESSION"
EOF

chmod +x ~/ns-allinone-3.40/ns-3.40/tcp_rl_project/run_pipeline_tmux.sh

# 3. Launch the new pipeline
cd ~/ns-allinone-3.40/ns-3.40/tcp_rl_project
./run_pipeline_tmux.sh