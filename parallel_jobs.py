""" Script to run multiple jobs in parallel across several GPUs """

import os, datetime, time
from numpy import floor
from subprocess import call

if __name__ == "__main__":

    # Collect system args
    data_type = sys.argv[1]
    trials = int(sys.argv[2])
    dimensions = int(sys.argv[3])
    epochs = int(sys.argv[4])
    samples = int(sys.argv[5])
    device = int(sys.argv[6])
    jobs_per_gpu = int(sys.arv[7])

    hidden_dims = [
                    '16',
                    '32',
                    '64',
                    '128',
                    ]

    batch_sizes = [
                  # '128',
                  # '256',
                  # '512',
                  '1024',
                  ]

    learning_rates = [
                      '2e-1',
                      '2e-2',
                      '2e-3',
                      ]
                      
    # Hyperparam search
    for hdim in hidden_dims:
        for lr in learning_rates:
            for bsize in batch_sizes:

                # Only allow certain number of jobs per GPU
                device += (1/jobs_per_gpu)

                # TMUX session name
                tmux_name = 'GPU{7}-{0}-{1}-samples-{2}-dims-{3}-{4}-{5}-{6}'.format(data_type, samples, dimensions,
                                                                              trials, lr, hdim, bsize, int(floor(device)))
                # Launch TMUX session
                call(['tmux', 'new', '-d', '-s', tmux_name])

                for trial in range(1, int(trials)+1):

                    # Get time
                    start_time = datetime.datetime.now().strftime("%Y-%m-%d-%s")

                    # Sent the job to that session
                    call(['tmux', 'send', '-t', tmux_name+'.0',
                          "CUDA_VISIBLE_DEVICES={0} ".format(int(floor(device))),
                          "python3 ", "parallel_main.py ",
                          data_type, ' ', '1', ' ', dimensions, ' ', hdim,
                          ' ', epochs, ' ', samples, ' ', bsize, ' ', lr, ' ', start_time+str(trial),
                          'ENTER'])

                # Send another command to kill the tmux session once it's done running
                # (easier to track progress using 'tmux ls')
                call(['tmux', 'send', '-t', tmux_name+'.0',
                      'tmux kill-session ', '-t ', tmux_name, 'ENTER'])
