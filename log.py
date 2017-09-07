# Author: Annus Zulfiqar 
# a small logger class for outputing text in python

class log(object):
    def __init__(self):
        pass
    def log(self, string=None, clause='log', end='\n', cute=False):
        if not string: 
            print('')
            return
        verbose = clause+': '+string if clause is not None and clause is not '' else string
        if cute:
            print('\b'*len(verbose), end='', flush=True)
            print(verbose, end='')
            return
        print(verbose, end=end) 
