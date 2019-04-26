import fileinput
import sys

strnums = sys.stdin.read().split(',')

title = ''

def evaluate(x):
    global title

    ret = isinstance(x, float)
    if not ret:
        title = x[:]
    return ret

def isnum(n):
    try:
        float(n)
        return True
    except ValueError:
        return False

buff = list(filter(
    lambda x: evaluate(x),
    [float(s) \
     if isnum(s) \
     else s \
     for s in strnums]))

if len(buff) > 0:
    avg = 0.0
    nmin = float('inf')
    nmax = -float('inf')

    for n in buff:
        if n < nmin:
            nmin = n
        if n > nmax:
            nmax = n
        avg += n

    avg /= float(len(buff))

    print('%s:%f,%f,%f' % (title, avg, nmin, nmax))
else:
    print('error:0.0,0.0,0.0')
