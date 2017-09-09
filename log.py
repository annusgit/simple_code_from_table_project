# Author: Annus Zulfiqar 
# a small logger function to output text in python

def log(string=None, clause='log', cute=False):
    if not string:
        print('')
        return
    verbose = clause+': '+string
    if not cute:
        print(verbose)
        return
    print('\b'*len(verbose), end='', flush=True)
    print(verbose, end='')

