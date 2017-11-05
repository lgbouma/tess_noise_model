# tess_noise_model
Sat 04 Nov 2017 05:04:53 PM EDT
Author: Luke Bouma

I took the following steps to generate the representative noise model.

Run `generate_bigfrac.py`

  Substeps I did for sanity checks:
    * plot_supersampled_PSFs.py
    * plot_att_err.py
    * plot_PRF_comparison.py

Run `ctd_avg_field_angle_avg.py`.

  This has a built-in sanity check: it writes the PRF_vs_temperature.pdf plot.

Run `noise_model.py`

Run `plot_noise_model.py`
