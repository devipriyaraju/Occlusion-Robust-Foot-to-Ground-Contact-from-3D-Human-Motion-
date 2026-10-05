# Compute Canada artifacts to preserve

The repository can run from raw AMASS and SMPL-H assets, but the following artifacts are useful for reproducing the exact reference run without recomputing every stage.

## Upload for exact result preservation

From `~/scratch/contact_project/`:

```text
results/
data_smplh_v2_T64_fps30_None.pt
```

The most useful files inside `results/` are:

```text
mlp_full.pt
tcn_foot.pt
tcn_full_novis.pt
tcn_full.pt
tcn_full_clean.pt
tcn_foot_seed1.pt
tcn_foot_seed2.pt
tcn_full_novis_seed1.pt
tcn_full_novis_seed2.pt
tcn_full_seed1.pt
tcn_full_seed2.pt
tcn_full_flatlabels.pt
contact_robustness.csv
contact_auroc.csv
per_subject.csv
root_refinement.csv
root_refinement_tuned.csv
seed_variance.csv
label_reliability.csv
contact_subsets.csv
labels_flat_vs_support.csv
contact_robustness.png
refinement_example.png
worst_subject_labels.png
labels_flat_vs_support.png
```

If there are job scripts, environment files, or scheduler logs, also preserve:

```text
*.sbatch
*.sh
requirements*.txt
environment*.yml
slurm-*.out
```

## Do not commit licensed assets

Do not publish these files in the GitHub repository:

```text
~/scratch/contact_project/amass/
~/scratch/contact_project/smplh/
~/scratch/CMU.tar.bz2
~/scratch/KIT.tar.bz2
~/scratch/smplh.tar.xz
```

They can be copied locally for reproduction, but they should remain ignored by Git.

## Missing physics stage

The current reference run contains contact prediction and contact-driven root correction. It does not contain joint torque estimation or ground reaction force estimation. If those experiments exist elsewhere on the cluster, preserve the corresponding source code, checkpoints, cached tensors, metrics, and figures so they can be added as a separate physics module instead of inferred from contact labels.
