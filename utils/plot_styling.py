"""Some plot styling utilities"""
from matplotlib import pyplot as plt
import seaborn as sns

COLORS = sns.color_palette('colorblind')

# Order legends using these
MODEL_ORDER = ['random', 'lr', 'Q1', 'Q2', 'Q3', 'hedonic-Q', 'state-Q', 'state-Q2', 'state-Q3', 'state-Q-cur', 'state-Q-C2S', 'state-Q-C2D', 'random_uni', 'hedonic-Q_uni', 'state-Q_uni', 'state-Q_uni_ad']
AEST_ORDER = ['benford', 'entropy', 'global_contrast_factor', 'symm', 'fd_aesthetics', 'complexity']
AEST_SHORT_LABELS = ['BLW', 'ENT', 'GCF', 'SYM', 'FRD', 'CPX']

AEST_ORDER_NEW = ['entropy', 'complexity']
AEST_SHORT_LABELS_NEW = ['ENT', 'FRD']

# Figure size for basic plots.
BASE_FIG_SIZE = (4, 2.5)

# Generic styles for learning schemes
# Usage:
# style = MODEL_STYLES['random']
# plt.plot(x, y, style['line style'], dashes=style['dashes'], color=style['color'], label=style['label'])
#
# Designed to be used with dictionary information where dict keys are model
# names in our experiments, e.g. Q1. (Labels will be the same as in our current paper.)
# See pstats.py create_collab_partner_plot for example.
MODEL_STYLES = {
    'random':
        {'line style': ':',
         'label': 'Random',
         'color': COLORS[0],
         'dashes': [2, 2],
         },
    'random_uni':
        {'line style': ':',
         'label': 'Random (unif.)',
         'color': COLORS[0],
         'dashes': [2, 2, 1, 1],
         },
    'Q1':
        {'line style': '-',
         'label': 'Direct-Q',
         'color': COLORS[1],
         'dashes': []  # Continuous line
         },
    'Q2':
         {'line style': '--',
          'label': 'Hedonic-Q',
          'color': COLORS[2],
          'dashes': [5, 2]
          },
    'Q3':
         {'line style': '--',
          'label': 'Altruistic-Q',
          'color': COLORS[3],
          'dashes': [5, 2, 2, 2]
          },
    'lr':
         {'line style': '--',
          'label': 'Linear reg.',
          'color': COLORS[5],
          'dashes': [5, 2, 1, 2, 1, 2]
          },
    'hedonic-Q':
        {'line style': '--',
         'label': 'Hedonic-Q',
         'color': COLORS[2],
         'dashes': [5, 2]
         },
    'hedonic-Q_uni':
        {'line style': '--',
         'label': 'Hedonic-Q (unif.)',
         'color': COLORS[2],
         'dashes': [5, 2, 2, 2]
         },
    'state-Q':
        {'line style': '-',
         'label': 'state-Q',
         'color': COLORS[1],
         'dashes': []  # Continuous line
         },
    'state-Q_uni':
        {'line style': '-',
         'label': 'state-Q (unif.)',
         'color': COLORS[2],
         'dashes': []  # Continuous line
         },
    'state-Q_uni_ad':
        {'line style': '-',
         'label': 'state-Q (unif. ad.)',
         'color': COLORS[3],
         'dashes': []  # Continuous line
         },
    'state-Q2':
         {'line style': '--',
          'label': 'state-Q (dyn)',
          'color': COLORS[3],
          'dashes': [5, 2, 2, 2]
          },
    'state-Q3':
         {'line style': '--',
          'label': 'state-Q (dyn) 2',
          'color': COLORS[5],
          'dashes': [5, 2, 1, 2, 1, 2]
          },
    'state-Q-cur':
        {'line style': '--',
         'label': 'state-Q (cur. old)',
         'color': COLORS[3],
         'dashes': [5, 2, 2, 2]
         },
    'state-Q-C2S':
        {'line style': '--',
         'label': 'state-Q (cur. static)',
         'color': COLORS[5],
         'dashes': [5, 2, 1, 2, 1, 2]
         },
    'state-Q-C2D':
        {'line style': '--',
         'label': 'state-Q (cur. dist)',
         'color': COLORS[4],
         'dashes': [5, 2, 5, 1, 1, 1]
         },
}

# Generic styles for different aesthetics
AEST_STYLES = {
    'benford':
        {'line style': ':',
         'label': "Benford's law",
         'color': COLORS[0],
         'dashes': [2, 2],
         },
    'entropy':
        {'line style': '-',
         'label': 'Entropy',
         'color': COLORS[1],
         'dashes': []  # Continuous line
         },
    'global_contrast_factor':
        {'line style': '--',
         'label': 'GCF',
         'color': COLORS[2],
         'dashes': [5, 2]
         },
    'symm':
        {'line style': '--',
         'label': 'Symmetry',
         'color': COLORS[3],
         'dashes': [5, 2, 2, 2]
         },
    'fd_aesthetics':
        {'line style': '--',
         'label': 'Fractal dimension',
         'color': COLORS[5],
         'dashes': [5, 2, 1, 2, 1, 2]
         },
    'complexity':
        {'line style': '--',
         'label': 'Fractal dimension',
         'color': COLORS[5],
         'dashes': [5, 2, 1, 2, 1, 2]
         },
}