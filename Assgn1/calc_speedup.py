import fileinput
import sys
import math

speeds = eval(sys.stdin.read())

print("Received: %s of %s" % (type(speeds), str(speeds)))

speedups = []

for t in speeds:
    speedup = t[0] / t[1]
    speedups.append((speedup, t[0], t[1], math.ceil(speedup)))

sspeedups = sorted(speedups, key=lambda s: s[0])

def vstring(ltuples, index):
    s = 'c('
    i = 0
    c = len(ltuples)
    for t in ltuples:
        s += str(t[index])
        if i < c - 1:
            s += ','
        i = i + 1
    s += ')'
    return s

print("speedup: %s" % vstring(sspeedups, 0))
print("speedup (ceiled): %s" % vstring(sspeedups, 3))
print("serial: %s" % vstring(sspeedups, 1))
print("parallel: %s" % vstring(sspeedups, 2))
