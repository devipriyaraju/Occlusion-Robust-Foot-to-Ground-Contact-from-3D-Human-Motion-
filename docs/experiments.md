# Experiment Artifact Index

This document maps the completed run artifacts included with the repository. The files under `results/tables/reference`, `results/figures/reference`, and `results/checkpoints/reference` were preserved from the supplied project outputs.

## Contact prediction and labels

- `contact_robustness.csv` contains controlled masking and corruption results.
- `contact_auroc.csv` contains contact ranking metrics.
- `contact_subsets.csv` contains subset-level contact results.
- `contact_v2_labels.csv` contains the later label evaluation.
- `labels_flat_vs_support.csv` compares flat-floor and support-aware labels.
- `label_reliability.csv` records label reliability analysis.
- `seed_variance.csv` records multi-seed stability.
- `per_subject.csv` and `loso_all.csv` contain subject-level generalization results.

## Dense contact

- `dense_contact.csv` compares dense per-vertex contact against four-point contact.
- `dense_contact_map.png` visualizes dense contact regions.
- `dense_tcn.pt` is the corresponding trained artifact.

## Motion reconstruction and tracking

- `reconstruction.csv` contains masked-joint reconstruction metrics.
- `balance_reconstruction.csv` measures balance consistency after reconstruction.
- `tracking_conditions.csv`, `tracking_long.csv`, and `tracker_chain.csv` contain tracking evaluations.
- `recon.pt`, `recon_bva.pt`, and `recon_bva_slide.pt` are trained reconstruction artifacts.

## 2D-to-3D lifting

- `lifting.csv` contains MPJPE and downstream contact F1 for one-view and two-view lifting.
- `prediction_2d_to_3d.png` and `dark_to_3d.png` visualize lifting examples.
- `lift_frame.pt`, `lift_tcn.pt`, and `lift_tcn_2view.pt` are trained lifting artifacts.

## Root refinement

- `refinement.csv`, `root_refinement.csv`, and `root_refinement_tuned.csv` contain contact-driven root correction results.
- `refinement_example.png` visualizes a refined trajectory.

## Ground reaction force and contact validation

- `physics_grf.csv` contains force-allocation and physical-consistency diagnostics.
- `contact_vs_force_plates.csv` compares motion-derived contact against measured contact over 144,472 frames.
- `physics_grf_example.png` visualizes a GRF example.

## Inverse dynamics and joint moments

- `torque_prediction.csv` compares a training-mean baseline, a single-frame model, and a temporal model for GRF and lower-body joint moments.
- `inverse_dynamics_summary.csv` reports residuals and sagittal ankle, knee, and hip moment ranges across motion classes.
- `full_body_dynamics.csv` contains full-body residual force and moment diagnostics.
- `photo_forces.csv` and `photo_forces.png` preserve the static-image force and moment experiment outputs.

## Balance

- `balance.csv` groups support and center-of-mass diagnostics by motion speed.
- `balance_reconstruction.csv` measures balance violation for reconstruction variants.

## Additional artifacts

The reference folders also contain extension experiments, causal and offline temporal variants, dynamics models, force-balance variants, figures, GIFs, and trained checkpoints. They are retained with their original filenames so the result history remains traceable.
