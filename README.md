# SCENARIO-BASED TESTING OF MARITIME COLLISION AVOIDANCE SYSTEMS

*Does an autonomous ship that follows the law stay safe and can we find the exact scenario where it doesn't?*

- 1,200+ baseline encounters
- real YOLOv8 detection on real imagery
- conformal safety guarantees   ·   
- CMA-ES / Bayesian falsification

This repository is an independent re-implementation, written from scratch, of an approach first explored during a summer student project. That project produced a prototype adapting a probabilistic scenario description language originally built for autonomous road vehicles, to the maritime domain.

> **No code from the original project is reproduced here.**
> It was never available to the author. Everything in this repository - the ship dynamics model, all five collision-avoidance strategies, the COLREGS compliance scoring, the perception pipeline, the uncertainty quantification, and the falsification search - is developed independently, that goes well beyond the original prototype's scope.

# Phase 1-The Baseline Harness

- Every scenario's initial geometry is solved backwards from its desired closest-approach distance and time, exact to 1×10⁻¹² m - no scenario is degenerate by construction.
- Ship maneuvering dynamics (a first-order Nomoto model) validated against IMO Res. MSC.137(76): all 5 hull types pass advance and tactical-diameter criteria.
- Safety and compliance trade off directly: the safest strategy (MPC) is 99.3% safe but only 0.47 compliant; the most compliant strategy (rule-based COLREGS, 0.68) is only 57.1% safe in the one situation the Rules require it to hold course rather than maneuver.
- Scenario-space coverage is measured, not asserted: 88% of reachable bearing×heading configurations, 100% of encounter×role combinations.



# Phase 2 - Perception in the Loop

- A YOLOv8 detector fine-tuned on the Singapore Maritime Dataset (4,455 object instances, evaluated across 5 scales) feeds a first-principles monocular ranging model - spherical-Earth horizon geometry, hull-down occlusion, dual-cue inverse-variance fusion — and an EKF/UKF tracker.
- Perception costs 33–44 percentage points of safety relative to a perfect-information control condition, run on identical scenarios and seeds.
- Mean time to a usable velocity estimate: 20.0 minutes, leaving only 3.8 minutes of time-to-closest-approach when the track becomes trustworthy - directly eroding the ‘ample time’ COLREGS Rule 8(a) assumes a human officer has.
- A segmentation model trained for horizon detection was evaluated against ground truth and found systematically biased (28.91 px, 100% of the error a constant offset) - traced to a real sensor/domain mismatch (coastal land horizon vs. open-water sea horizon) - and correctly rejected by a principled invertibility check rather than silently used.



# Phase 3 - Uncertainty, Calibration and Legal Role Inversion

- A 5-member perception ensemble decomposes error into aleatoric (irreducible) and epistemic (model-reducible) components - 26% of range variance is epistemic, i.e. addressable with a better model.
- The raw uncertainty estimate is miscalibrated (ECE 0.138). Split conformal prediction restores a distribution-free guarantee: 89.0% empirical coverage at a 90% target, with no Gaussian assumption.
- A risk-aware CAS acting on this bound was found, diagnosed, and fixed to be less safe than the point-estimate CAS when acting before the track converges - uncertainty-awareness must include knowing when the uncertainty estimate is itself meaningless.
- COLREGS Rule 18 makes legal right-of-way depend on vessel category. Using CLIP zero-shot classification for categories no maritime dataset labels, misclassification is shown to invert a vessel's legal obligation - 5.6× more often in the dangerous direction than the merely conservative one.



# Phase 4 - Falsification and Deployment

- Uniform random sampling is replaced with directed search (CMA-ES, Bayesian optimization) over the scenario space. Random search found a worst-case separation of −134 m in 200 trials; CMA-ES (multi-restart) found −371 m; Bayesian optimization found −578 m — a genuine near-collision at 2 m clearance.
- A behavior-cloned neural policy, distilled from an MPC expert, achieves excellent imitation accuracy (MSE 0.00137) yet its closed-loop safety collapses from 99.4% to 47.2% — imitation accuracy does not predict deployed safety.
- That policy was evaluated adversarially: directed search exposed a near-total collision (ρ = −924 m) that random testing - the default validation protocol for most learned systems — reported as only ρ = −605 m, a 319 m blind spot standard testing would have missed.
- The distilled policy runs 995× faster than the MPC expert - a deployable control policy, at a safety cost that is measured, not assumed away.



# Architecture

```
camera frame ──▶ YOLOv8 detection ──▶ monocular bearing/range ──▶ EKF/UKF tracker
                                                                        │
                                                                        ▼
scenario space ──▶ Scenic / analytic sampler ──▶ ship dynamics ──▶ estimated target state
      ▲                                                                │
      │                                                                ▼
      └── CMA-ES / Bayesian falsification ◀── robustness score ◀── CAS decision ──▶ COLREGS scoring
                                                                        │
                                                                        ▼
                                                          conformal uncertainty bound
```

# Related Work

- Fremont et al., Scenic: A Language for Scenario Specification and Scene Generation, PLDI 2019.
- Dreossi et al., VerifAI: A Toolkit for the Formal Design and Analysis of AI-Based Systems, CAV 2019.
- IMO COLREGS (1972, as amended) and IMO Res. MSC.137(76).
- Vovk, Gammerman & Shafer, Algorithmic Learning in a Random World (conformal prediction).
- Radford et al., Learning Transferable Visual Models From Natural Language Supervision (CLIP), ICML 2021.

# Datasets and Pretrained Models

Phase 2’s detector is fine-tuned on real maritime imagery, and Phase 2/4's horizon extraction is trained on a real segmentation dataset. Both carry citation requirements from their creators, reproduced here in full:

- Singapore Maritime Dataset. D. K. Prasad, D. Rajan, L. Rachmawati, E. Rajabally, and C. Quek, "Video Processing from Electro-Optical Sensors for Object Detection and Tracking in a Maritime Environment: A Survey," IEEE Transactions on Intelligent Transportation Systems, vol. 18, no. 8, pp. 1993–2016, 2017.
- MaSTr1325. B. Bovcon, J. Muhovič, J. Perš, and M. Kristan, "The MaSTr1325 Dataset for Training Deep USV Obstacle Detection Models," in 2019 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2019, pp. 3431–3438.
- Ultralytics YOLOv8. G. Jocher, A. Chaurasia, and J. Qiu, Ultralytics YOLOv8 (Version 8.0.0) [Software], 2023. Available: github.com/ultralytics/ultralytics.

*Neither dataset's images or labels are redistributed in this repository - only the fitted model artefacts and the code that trains them from a fresh download.*

# Limitations

- Perception is measured on real imagery but simulated in the closed loop via a fitted error model, not a live detector, for tractability - stated explicitly wherever it applies.
- Encounters are pairwise; multi-vessel situations are not yet modelled.
- The learned policy is a compact MLP trained by behavior cloning, not a state-of-the-art architecture - its failure modes are a lower bound on what a more capable learned controller might also exhibit.

# Citation

```
@software{das_maritime_cas,
  author = {Das, Atanu},
  title  = {Scenario-Based Testing of Maritime Collision Avoidance Systems},
  year   = {2025},
  url    = {https://github.com/atanu-dip/maritime-cas-verification}
}
```

# Acknowledgements

The initial idea for this project - adapting scenario description language to maritime collision avoidance - originated and supervised within DNV's Research and Development unit.
