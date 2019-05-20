import os

counts = [4, 8, 12, 16]
sims = 5
time = 120
prefix = 'wp'

avgs = []

def ccount(n):
    (n / 2) - 1

for num in counts:
    s = 0
    for sim in range(1, sims + 1):
        folder = prefix + '_sim' + str(sim)
        folder2 = prefix + '_log' + str(num)
        fname = prefix + '_output_' + str(num) + '_' + str(time) + '.log'
        path = os.path.join(folder, folder2, fname)
        with open(path, 'r') as f:
            b = int(f.read().strip().split(':')[1])
            s += b / time
            #print('from %s\n\t%s' % (path, f.read()))
    s /= sims
    avgs.append(s)
for i in range(len(counts)):
    print('For n = %i: %d messages per second' % (counts[i], avgs[i]))
