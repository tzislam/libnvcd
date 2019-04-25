import sys, os

stdin = sys.stdin.read().split(':')
base = stdin[0]

# newline is appended to the end of echo, so we just
# call strip here for every element (obv inefficient,
# but who cares)
dirs = [x.strip() for x in stdin[1].split('|')]

entries_max = []
entries_min = []
entries_avg = []

def print_entries(etype, e):
    s = etype + '\n'
    for x in e:
        s += '\t{0}:{1}\n'.format(x[0], x[1])
    print(s)

for idir in dirs:
    path = os.path.join(base, idir, 'amm')

    with open(path, 'r') as f:
        read_data = f.read().strip()

    splt = read_data.split(':')
    title = splt[0]
    amm = splt[1].split(',')

    entries_avg.append((title, amm[0]))
    entries_min.append((title, amm[1]))
    entries_max.append((title, amm[2]))

print_entries('avg', entries_avg)
print_entries('min', entries_min)
print_entries('max', entries_max)
