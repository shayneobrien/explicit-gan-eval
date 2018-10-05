import os, datetime, time
from numpy import floor
from subprocess import call

data_type = 'multivariate'
trials = '8'
dimensions = '16'
epochs = '25'
samples = '10000'

device = 0
jobs_per_gpu = 8

for trial in range(1, int(trials)+1):

    # Only allow certain number of jobs per GPU
    device += (1/jobs_per_gpu)

    # Name GPU
    tmux_name = 'GPU{0}-{1}-{2}-samples-{3}-dims-{4}'.format(int(floor(device)),
                                                             data_type,
                                                             samples,
                                                             dimensions,
                                                             trial)

    # Launch session
    call(['tmux', 'new', '-d', '-s', tmux_name])

    # Sent the job to that session
    call(['tmux', 'send', '-t', tmux_name+'.0',
          "CUDA_VISIBLE_DEVICES={0} ".format(int(floor(device))),
          "python3 ", "main.py ",
          data_type, ' ', '1', , dimensions, ' ', hdim,
          ' ', epochs, ' ', samples,
          'ENTER'])

    call(['tmux', 'send', '-t', tmux_name+'.0',
          'kill-session ', '-t ', tmux_name, 'ENTER'])

    time.sleep(1.02)
