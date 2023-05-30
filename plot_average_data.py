import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_file = pd.read_csv("/scf-data/kopal/3lakh_intron_exon/3lakh_tri_es/IE_3lakhstart_bbone_ang.csv")

IE_dataframe = pd.DataFrame(data_file)
avg_data = list(IE_dataframe.mean(axis=0))
del avg_data[0]
x = list(np.arange(1, 376, 1))


plt.plot(x, avg_data, linewidth=2)
plt.savefig('3lakh_ES_tri_bbone_ang_plot.png')



