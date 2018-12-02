""" Script to run full study as in paperself. Please note that usage of this
script is not recommended as it would take more than a GPU year to complete.

Instead, see parallel_jobs.py and job.sh for ways to parallelize our code
"""
import os
from subprocess import call

if __name__ == '__main__':

    for trials in ['20']:
        for epochs in ['25']:
            for samples in ['1000', '10000', '100000']:
                for dimensions in ['16', '32', '64', '128']:
                    call(["python3", "main.py", "multivariate",
                          trials, dimensions, epochs, samples])
