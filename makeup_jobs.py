import os, datetime, time
from numpy import floor
from numpy.random import randint
from subprocess import call

hyperparams = [('2e-01', '32', '1024'),
 ('2e-02', '32', '128'),
 ('2e-02', '64', '128'),
 ('2e-01', '32', '128'),
 ('2e-01', '32', '1024'),
 ('2e-01', '64', '128'),
 ('2e-02', '32', '128'),
 ('2e-02', '64', '128')]


data_type = 'multivariate'
trials = '1'
dimensions = '16'
epochs = '25'
samples = '100000'

device = 5
jobs_per_gpu = 16

# Hyperparam search
for lr, hdim, bsize in hyperparams:

    # Only allow certain number of jobs per GPU
    device += (1/jobs_per_gpu)

    if int(floor(device)) == 8:
        dev2 = 0

        # TMUX session name
        tmux_name = 'GPU{7}-{0}-{1}-samples-{2}-dims-{3}-{4}-{5}-{6}'.format(data_type, samples, dimensions,
                                                                            trials, lr, hdim, bsize, int(floor(dev2)))
        # Get time
        start_time = datetime.datetime.now().strftime("%Y-%m-%d-%s")

        call('tmux', 'send', '-t', tmux_name+'.0', 'bash', 'ENTER')

        # Sent the job to that session
        call(['tmux', 'send', '-t', tmux_name+'.0',
              "CUDA_VISIBLE_DEVICES={0} ".format(int(floor(device))),
              "python3 ", "parallel_main.py ",
              data_type, ' ', '1', ' ', dimensions, ' ', hdim,
              ' ', epochs, ' ', samples, ' ', bsize, ' ', lr, ' ', start_time+str(randint(10000)),
              'ENTER'])

        dev2 += (1/jobs_per_gpu)

        if int(floor(dev2)) == 8:
            dev2 = 0

    # TMUX session name
    tmux_name = 'GPU{7}-{0}-{1}-samples-{2}-dims-{3}-{4}-{5}-{6}'.format(data_type, samples, dimensions,
                                                                        trials, lr, hdim, bsize, int(floor(device)))

    # Launch TMUX session
    call(['tmux', 'new', '-d', '-s', tmux_name])

    # Get time
    start_time = 'finish'

    # Sent the job to that session
    call(['tmux', 'send', '-t', tmux_name+'.0',
          "CUDA_VISIBLE_DEVICES={0} ".format(int(floor(device))),
          "python3 ", "parallel_main.py ",
          data_type, ' ', '1', ' ', dimensions, ' ', hdim,
          ' ', epochs, ' ', samples, ' ', bsize, ' ', lr, ' ', start_time+str(randint(100)),
          'ENTER'])

    # Send another command to kill the tmux session once it's done running
    # (easier to track progress using 'tmux ls')
    call(['tmux', 'send', '-t', tmux_name+'.0',
          'tmux kill-session ', '-t ', tmux_name, 'ENTER'])
