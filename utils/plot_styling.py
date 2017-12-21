"""Some plot styling utilities"""
from matplotlib import pyplot as plt
import seaborn as sns

COLORS = sns.color_palette('colorblind')

# Order legends using these
MODEL_ORDER = ['random', 'lr', 'Q1', 'Q2', 'Q3']
AEST_ORDER = ['benford', 'entropy', 'global_contrast_factor', 'symm', 'fd_aesthetics']
AEST_SHORT_LABELS = ['BLW', 'ENT', 'GCF', 'SYM', 'FRD']

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
}