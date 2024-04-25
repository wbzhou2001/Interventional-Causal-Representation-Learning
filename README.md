# Interventional Causal Representation Learning

Simple demo for [Interventional Causal Representation Learning by Ahuja et. al. 2022](https://arxiv.org/abs/2209.11924) affiliated for our CMU 10716 course project final report. 
Code is adapted [original code repository](https://github.com/facebookresearch/CausalRepID).

Please see the <tt>synthetic_data_exp.ipynb</tt> and <tt>real_data_exp.ipynb</tt> to reimplement our experiment results.
Our result includes causal representation learning over synethic data experiment uses PyGame generated images with hard do-interventions, as well as causal representation learning with deep brain stimulation real data.
Please rerun the notebooks to reimplement our experiment results.

![](figs/recon.png =250x)

We note that due to Github's file upload size limit we have cut the train size from 20000 to 1000 for our synthetic experiment, and took the first 200 features for our real data experiment in this repository.
