import os
from subprocess import call

if __name__ == '__main__':

    """
    sysargs:
    # (1) dataset: multivariate, mixture, circles, or mnist \n
    # (2) trials (for confidence intervals) 1 \n
    # (3) number of dimensions: 1, 10, 100, 1000, etc. \n
    # (4) number of epochs: 10, 100, 1000, etc. \n
    # (5) number of samples: 1000, 10000, 100000, etc. \n
    """

    # Running across 20 GPUs
    for trials in ['1']:
        for epochs in ['25']:

            call(["python3", "main.py", "mnist",
                  trials, '0', epochs, '0'])

            for samples in ['1000', '10000', '100000', '1000000']:

                call(["python3", "main.py", "circles",
                      trials, '0', epochs, samples])

                for dimensions in ['16', '32', '64', '128', '256', '512', '1024', '2048']:

                    call(["python3", "main.py", "multivariate",
                          trials, dimensions, epochs, samples])

                    for mixtures in ['1000', '10000', '100000', '1000000']:

                        call(["python3", "main.py", "mixture",
                              trials, dimensions, epochs, samples, mixtures])
