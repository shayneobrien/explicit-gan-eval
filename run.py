from subprocess import call

if __name__ == '__main__':

    # sysargs:
    # (1) dataset: multivariate, mixture, circles, or mnist \n
    # (2) trials (for confidence intervals) 1 \n
    # (3) number of dimensions: 1, 10, 100, 1000, etc. \n
    # (4) number of epochs: 10, 100, 1000, etc. \n
    # (5) number of samples: 1000, 10,000, 100,000, etc. \n

    for trials in [100]:
        for dimensions in ['16', '32', '64', '128', '256', '512', '1024', '2048']:
            for epochs in [25]:

                call(["python", "main.py", "multivariate",
                      trials, dimensions, epochs])

                call(["python", "main.py", "mnist",
                      trials, dimensions, epochs])

                # call(["python", "main.py", "circles",
                #       trials, dimensions, epochs])

                for samples in ['1000', '10000', '100000', '1000000']:

                    call(["python", "main.py", "mixture",
                          trials, dimensions, epochs, samples])
