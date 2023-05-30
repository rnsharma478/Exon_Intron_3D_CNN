#!/bin/bash
#PBS -S /bin/bash
#PBS -N 3lakh_plt_avg_2
#PBS -l select=1:ncpus=32
#PBS -l walltime=240:00:00
#PBS -q genomeq
#PBS -o 3lakh_plt_avg_2.txt
#PBS -e 3lakh_plt_avg_err_2.txt
#PBS -joe

# ## export environment variable from current session to job run-time ... better to use this always.
#PBS -V

#$$P@@@B&&&S -()(q workq

###########Write your command as you write it on command line. Then call "qsub msa.sh"##################
python /scf-data/kopal/3lakh_intron_exon/plot_average_data.py
