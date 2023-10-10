#!/bin/bash
session=calcHist

tmux new-session -d -s $session 'htop';         # start new detached tmux session, run htop

tmux split-window;
tmux send './calcHist.py --start=2009-01-01 --stop=2013-01-01' ENTER;

tmux split-window;
tmux send './calcHist.py --start=2013-01-01 --stop=2015-01-01' ENTER;

tmux select-layout even-vertical

tmux split-window -h -t 0;
tmux send './calcHist.py --start=2015-01-01 --stop=2017-01-01' ENTER;

tmux split-window -h -t 2;
tmux send './calcHist.py --start=2017-01-01 --stop=2019-01-01' ENTER;

tmux split-window -h -t 4;
tmux send './calcHist.py --start=2019-01-01 --stop=2020-01-01' ENTER;

tmux a;                                         # open (attach) tmux session.
