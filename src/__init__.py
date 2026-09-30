"""Shared pipeline for the fair re-evaluation (preregistration_v3).

Modules, in the order data flows through them:
    smd        load one SMD machine (train, test, test labels)
    splits     train -> fit part + validation (last 20%)
    windows    cut series into windows (60, step 10) and label test windows
    detector   Isolation Forest with fixed defaults (+ scaler fitted on training only)
    threshold  score threshold at 1% false alarms on validation windows
    metrics    PR-AUC, ROC-AUC, recall at threshold, false alarms per day
    bootstrap  95% cluster bootstrap CI over machines
    metadata   git commit, seed, library versions saved next to every result
    config     load one YAML config per experiment
    prepare    glue: one machine -> train / validation / test windows
"""
