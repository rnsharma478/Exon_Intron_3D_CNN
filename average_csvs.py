import pandas as pd
import numpy as np

file_ang = "G:\\3_lac_intron_exon_data\\3lakh_intron_exon\\3lakh_tri_ee\\IE_3lakhend_bp_ang_c.csv"
file_deg = "G:\\3_lac_intron_exon_data\\3lakh_intron_exon\\3lakh_tri_ee\\IE_3lakhend_bp_deg.csv"

file_one = pd.read_csv(file_ang)
file_two = pd.read_csv(file_deg)

file_one_df = pd.DataFrame(file_one)
file_two_df = pd.DataFrame(file_two)

df_avg = (file_one_df+file_two_df)/2
print(file_one_df.iloc[0])
print(file_two_df.iloc[0])
print(df_avg.iloc[0])
df_avg.to_csv("G:\\3_lac_intron_exon_data\\3lakh_intron_exo\\3lakh_tri_ee\\IE_3lakhend"
              "_bp.csv")

