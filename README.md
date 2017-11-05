# tess_noise_model

I took the following steps to generate the representative noise model.

Ran `generate_bigfrac.py`

  Substeps I did for sanity checks:
    * `plot_supersampled_PSFs.py`
    * `plot_att_err.py`
    * `plot_PRF_comparison.py

Ran `ctd_avg_field_angle_avg.py` This has a built-in sanity check: it writes
the `PRF_vs_temperature.pdf` plot.

Ran `noise_model.py`

Ran `plot_noise_model.py`
