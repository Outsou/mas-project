"""Some plot styling utilities"""
from matplotlib import pyplot as plt
import seaborn as sns

LINE_STYLES = ['-', '--', '-.', ':', '--.', '-..']
COLORS = sns.color_palette('colorblind')

MODEL_STYLES = {
    'random':
        {'line style': ':',
         'label': 'Random',
         'color': COLORS[0],
         },
    'Q1':
        {'line style': '-',
         'label': 'Simple-Q',
         'color': COLORS[1],
         },
    'Q2':
        {'line style': '--',
         'label': 'Hedonistic-Q',
         'color': COLORS[2],
         },
    'Q3':
        {'line style': '-.',
         'label': 'Altruistic-Q',
         'color': COLORS[3],
         },
    'lr':
        {'line style': '--.',
         'label': 'Linear reg.',
         'color': COLORS[4],
         },
}

AEST_STYLE = {
    'benford':
        {'line style': ':',
         'label': "Benford's law",
         'color': COLORS[0],
         },
    'entropy':
        {'line style': '-',
         'label': 'Entropy',
         'color': COLORS[1],
         },
    'global_contrast_factor':
        {'line style': '--',
         'label': 'GCF',
         'color': COLORS[2],
         },
    'symm':
        {'line style': '-.',
         'label': 'Symmetry',
         'color': COLORS[3],
         },
    'fd_aesthetics':
        {'line style': '--.',
         'label': 'Fractal dimension',
         'color': COLORS[4],
         },
}