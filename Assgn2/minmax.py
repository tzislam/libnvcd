import os

import sys

counts = [4, 8, 12, 16]
sims = 5
time = 120
prefix = sys.stdin.read().strip()

print("prefix: %s" % prefix)

mins = []
maxs = []

def ccount(n):
    (n / 2) - 1
    
for num in counts:
    smin = sys.maxsize
    smax = -sys.maxsize - 1
    
    for sim in range(1, sims + 1):
        folder = prefix + '_sim' + str(sim)
        folder2 = prefix + '_log' + str(num)
        fname = prefix + '_output_' + str(num) + '_' + str(time) + '.log'
        path = os.path.join(folder, folder2, fname)
        
        with open(path, 'r') as f:
            b = int(f.read().strip().split(':')[1])
            s  = b / time
            smax = s if s > smax else smax
            smin = s if s < smin else smin
            
    mins.append(smin)
    maxs.append(smax)
    
for i in range(len(counts)):
    print("For n = %i:\n\tmax %d messages per second\n\tmin %d messages per second\n"
          % (counts[i], maxs[i], mins[i]))
