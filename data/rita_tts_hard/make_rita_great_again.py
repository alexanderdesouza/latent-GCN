import pandas as pd
import numpy as np

df = pd.read_csv('rita_tts_hard.content', delimiter='\t', header=None)

del df[27]
del df[29]
del df[33]
del df[34]
del df[35]
del df[36]

df.to_csv('rita_tts_hard.content', sep='\t', index=False, header=False)
