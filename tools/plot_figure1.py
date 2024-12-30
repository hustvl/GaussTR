import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import LogLocator, MultipleLocator

sns.set_theme(
    style='ticks', rc={
        'axes.spines.right': False,
        'axes.spines.top': False
    })
sns.axes_style(rc={'axes.grid': True})

data = pd.DataFrame({
    'training duration': [-1, 1, 0.093, 2, -0.263],
    'performance (mIoU)': [111.7, 19.3, 19.53, 18.93, 91.94],
    'color':
    ['open-voc', 'closed-voc', 'closed-voc', 'open-voc', 'closed-voc'],
    'shape': ['GS', 'VR', 'VR', 'VR', 'GS']
})

plt.figure(figsize=(6, 5))
sns.scatterplot(
    data=data,
    x='training duration',
    y='performance (mIoU)',
    hue='color',
    style='shape',
    hue_order=['closed-voc', 'open-voc'],
    palette='Set1',
    markers=['o', 's'],
    legend=None,
    s=10)

# Set the color of spines
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_color('black')
plt.gca().spines['top'].set_color('black')
plt.gca().spines['right'].set_color('black')

# Set the color of ticks
plt.gca().tick_params(axis='x', colors='black')
plt.gca().tick_params(axis='y', colors='black')

# Set the color of grids
plt.grid(color='gray', linestyle='--', linewidth=0.4)

# Set the interval of ticks
plt.gca().xaxis.set_major_locator(MultipleLocator(1))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))

# Set the range of ticks
plt.xlim(-1.75, 2.75)
plt.ylim(8.5, 12.25)

# plt.show()
plt.savefig('fig.png', dpi=300)
