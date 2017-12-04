from __future__ import print_function
import tarfile
from tqdm import trange, tqdm
from collections import Counter

def termscale(freq):
    return 1.0 - 2 ** - freq

TERMS = 50

tar = tarfile.open('/Users/Peter/Dropbox/Datasets/interaction/wikitalk/comments_user_2004.tar.gz')

lines = tar.extractfile('comments_user_2004/chunk_0.tsv')

# spl[3] Timestamp
# print spl[5] Sender

line = lines.readline()
spl = line.split('\t')
print(line)

authors = set()
recipients = set()
words = Counter()

for line in tqdm(lines):
    spl = line.split('\t')
    # print 'author:', spl[7], 'recipient: ', spl[5].split('/')[0], 'time: ', spl[3], 'content:', spl[1][:100]
    authors.add(spl[7])
    recipients.add(spl[5].split('/')[0])
    for word in spl[1].split():
        words[word] += 1

print('authors ', len(authors))
print('recipients ', len(recipients))

print('most common words ', words.most_common(5))

auth_dict = {k: v for v, k in enumerate(authors)}
rec_dict = {k: v for v, k in enumerate(recipients)}

term_lst = [pair[0] for pair in words.most_common(TERMS)]

lines = tar.extractfile('comments_user_2004/chunk_0.tsv')
line = lines.readline()

with open('wikitalk.cites', 'w') as f:
    for line in tqdm(lines):
        spl = line.split('\t')

        author = spl[7]
        recipient = spl[5].split('/')[0]

        freqs = Counter()
        for term in spl[1].split():
            freqs[term] += 1

        bow = [termscale(freqs[word]) for word in term_lst]

        print('\t'.join(map(str, [auth_dict[author], rec_dict[recipient]] + bow)), file=f)









