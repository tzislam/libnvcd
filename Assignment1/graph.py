import sys, os
import matplotlib
import matplotlib.pyplot as plt

# plt is derived from https://matplotlib.org/gallery/statistics/barchart_demo.html

stdin = sys.stdin.read().split(':')
base = stdin[0]

# newline is appended to the end of echo, so we just
# call strip here for every element (obv inefficient,
# but who cares)
dirs = [x.strip() for x in stdin[1].split('|')]

entries_max = []
entries_min = []
entries_avg = []
entry_profiles = []

bar_count = 0

bar_width = 0.35
opacity = 1.0

def print_entries(etype, e):
    s = etype + '\n'
    for x in e:
        s += '\t{0}:{1}\n'.format(x[0], x[1])
    print(s)

def lrange(x):
    return list(range(len(x)))

def make_bar(ax, color, etype, e):
    global bar_count
    global bar_width

    xlist = [x for x in lrange(e)]
    ylist = [u[1] for u in e]
    
    bar_count += 1

    return ax.bar(
        xlist, ylist, bar_width,
        alpha=opacity,
        color=color,
        label=etype
    )

for idir in dirs:
    path = os.path.join(base, idir, 'amm')

    with open(path, 'r') as f:
        read_data = f.read().strip()

    splt = read_data.split(':')
    profile = splt[0]
    amm = splt[1].split(',')

    entry_profiles.append(profile)

    entries_avg.append((profile, amm[0]))
    entries_min.append((profile, amm[1]))
    entries_max.append((profile, amm[2]))

print_entries('avg', entries_avg)
print_entries('min', entries_min)
print_entries('max', entries_max)

fig, ax = plt.subplots()

bavg = make_bar(ax, 'b', 'avg', entries_avg)
#bmin = make_bar(ax, 'r', 'min', entries_min)
#bmax = make_bar(ax, 'g', 'max', entries_max)

ax.set_xlabel('Profiles')
ax.set_ylabel('Times')
ax.set_title('yeet')
ax.set_xticks([x for x in lrange(entries_avg)])
ax.set_xticklabels(entry_profiles)
ax.legend()

fig.tight_layout()
plt.show()
