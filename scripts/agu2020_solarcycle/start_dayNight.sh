#!/bin/bash
session=plot

tmux new-session -d -s $session 'htop';         # start new detached tmux session, run htop

tmux split-window;
tmux send './dayNight.py --data_source WSPRNet' ENTER;

tmux split-window;
tmux send './dayNight.py --data_source RBN' ENTER;

tmux split-window;
tmux send './dayNight.py --data_source WSPRNet_RBN' ENTER;

tmux select-layout even-vertical
tmux a;                                         # open (attach) tmux session.
