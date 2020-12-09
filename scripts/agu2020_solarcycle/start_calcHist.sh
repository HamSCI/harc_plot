#!/bin/bash
session=calcHist

tmux new-session -d -s $session 'htop';         # start new detached tmux session, run htop

tmux split-window;
tmux send './calcHist_1.py' ENTER;

tmux split-window;
tmux send './calcHist_3.py' ENTER;

tmux select-layout even-vertical

tmux split-window -h -t 0;
tmux send './calcHist_0.py' ENTER;

tmux split-window -h -t 2;
tmux send './calcHist_2.py' ENTER;

tmux split-window -h -t 4;
tmux send './calcHist_4.py' ENTER;

tmux a;                                         # open (attach) tmux session.
