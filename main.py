from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

seeds_df = pd.read_csv("http://qps.ru/jNZUT")

# Удалиние информации об образцах зерна
varieties = list(seeds_df.pop('grain_variety'))

samples = seeds_df.values


hierarchical_clustering = linkage(samples, method='complete')

# Дендограмма
dendrogram(hierarchical_clustering,
           labels=varieties,
           leaf_rotation=90,
           leaf_font_size=6,
           )

plt.show()
