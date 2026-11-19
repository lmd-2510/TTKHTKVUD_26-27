# Single-Trial EEG Classification in the Oddball Task

## Dataset and Data Acquisition

### Dataset and recording selection

The Nencki–Symfonia EEG/ERP dataset (OpenNeuro accession `ds004621`) contains recordings from 42 healthy adults across three experimental tasks (Oddball, MSIT+, and SRT) and a resting-state protocol. The study used a 128-electrode cap. The Oddball recordings were sampled at 1,000 Hz, and the downloaded channel tables list **127 EEG channels per participant**. The initial experiments will use the recorded EEG channels, subject to channel quality checks during preprocessing. [1–3]

We selected the Oddball recordings because the research question asks whether a single EEG segment can distinguish a frequent standard stimulus from a rare target stimulus. The Oddball protocol also presented rare distractors. Participants were instructed to press a button when a target appeared. The task metadata gives approximate stimulus proportions of 76% standard, 12% target, and 12% distractor. The download excluded MSIT+, SRT, resting-state recordings, participant metadata tables, and auxiliary study files because the binary classification uses the Oddball EEG and stimulus markers. [1–3]

### Files and provenance

We retrieved 213 files for 42 participants (27.19 GiB) from the public OpenNeuro mirror. The download contains five files per participant and three shared metadata files. The acquisition manifest records each file's source, size, checksum, and the per-participant event and channel counts. [2,4]

| File | Role in the analysis |
|---|---|
| `*_eeg.eeg` | Continuous EEG samples in BrainVision binary format. |
| `*_eeg.vhdr` | BrainVision header used to open the recording in MNE. It specifies the signal format and references the signal and marker files. |
| `*_eeg.vmrk` | BrainVision event markers at sample positions. |
| `*_events.tsv` | Event table. The `onset` field gives event time in seconds and `event_type` identifies the event. |
| `*_channels.tsv` | Channel inventory used to identify EEG channels and verify the recorded channel count. |
| `task-oddball_eeg.json` | Shared description of the Oddball protocol and acquisition settings. |
| `task-oddball_events.json` | Shared dictionary for the event table and its codes. |
| `dataset_description.json` | Dataset identification and provenance. |

### Event codes and candidate trials

One standard or target onset defines one candidate trial. We use the stimulus onset in `events.tsv` as time zero for epoching and use the stimulus code to assign the class. Button presses are recorded separately. A target retains its target label even if the participant did not press the button. The downloaded continuous EEG and event tables preserve all event types. [3,4]

| Event code | Meaning | Raw event rows | Use |
|---|---|---:|---|
| `S 5` | Standard or frequent stimulus | 21,018 | Class 0 |
| `S 6` | Target stimulus | 3,422 | Class 1 |
| `S 7` | Distractor or deviant stimulus | 3,425 | Excluded from the binary sample set |
| `S 1` | Recorded button response | 3,872 | Retained as event metadata |
| `S 10` | Break marker | 168 | Retained as event metadata |
| `S 12` | Beginning-of-run marker | 125 | Retained as event metadata |

The inventory also lists one entry each for `R 15`, `S 14`, and `S 15`, plus two entries reported as `empty`. The binary label mapping uses only `S 5` and `S 6`. The downloaded event tables contain **24,440 candidate binary trials**, of which **86.0% are standard**. These counts precede epoch validation and artifact rejection. The final sample counts and class distribution will be reported after preprocessing. [4]

## References

1. Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. The Nencki-Symfonia electroencephalography/event-related potential dataset. *GigaScience*. 2022. [doi:10.1093/gigascience/giac015](https://doi.org/10.1093/gigascience/giac015).
2. Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. Nencki–Symfonia EEG/ERP dataset. [OpenNeuro ds004621](https://openneuro.org/datasets/ds004621). [GigaDB dataset DOI](https://doi.org/10.5524/100990).
3. OpenNeuroDatasets. [Oddball acquisition metadata](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/main/task-oddball_eeg.json) and [event dictionary](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/main/task-oddball_events.json).
4. Project data acquisition inventory. [Oddball EEG dataset and acquisition](data_acquisition.pdf), p. 2.

## Preprocessing reporting notes

The preprocessing section will describe the input data, the processing steps and their rationale, the resulting data, and observed problems with the actions taken to investigate them.

During the initial sub-01 run, amplitude-based rejection across 127 EEG channels retained 175/498 standards and 12/81 targets. Fp2 exceeded the 150 uV peak-to-peak threshold in 391/579 candidate epochs. No channel fell below the 0.1 uV flatness threshold in the diagnostic table. After reviewing and excluding ICA component 0, the same thresholds retained 496/498 standards and 81/81 targets. The subsequent 16-channel run retained the same counts. These are pilot findings.

Suggested figures are the channel-wise threshold violation counts, the sub-01 ICA component 0 trace alongside Fp1/Fp2, and the sub-02 EEG overlay before and after ICA at Fp1, Fp2, Fz, Cz and Pz. The sub-02 overlay covers 30-90 seconds and illustrates reduced frontal transients. Final cohort counts must use epochs regenerated from the ICA-cleaned continuous recordings.

The 150 uV threshold is an initial amplitude rejection choice. Published ERP protocols have used 150 uV moving-window peak-to-peak criteria; our implementation computes peak-to-peak over the entire epoch (-200 to 1000 ms), so those criteria are not equivalent. The 0.1 uV minimum peak-to-peak threshold was an engineering choice for near-flat epochs and has not been established here as a dataset-specific published recommendation. Neither threshold is a mandatory MNE default.

References for these notes are the [MNE Epochs documentation](https://mne.tools/stable/generated/mne.Epochs.html) and the [published ERP protocol with a 150 uV moving-window criterion](https://bio-protocol.org/exchange/minidetail?id=19622089&type=30).

### Evidence for amplitude thresholds

Navid MS, Niazi IK, Lelic D, Drewes AM, Haavik H (2019). The Effects of Filter’s Class, Cutoff Frequencies, and Independent Component Analysis on the Amplitude of Somatosensory Evoked Potentials Recorded from Healthy Volunteers. Sensors 19(11), 2610. DOI https://doi.org/10.3390/s19112610. Their artifact rules include a 150 uV peak-to-peak criterion within 200 ms sliding windows stepped by 100 ms. This is supporting precedent from a somatosensory ERP study. Our whole-epoch (-200 to 1000 ms) criterion is a project adaptation. The paper also uses other artifact criteria, so its pipeline is not reproduced by the single 150 uV threshold. Source PDF https://vbn.aau.dk/files/305686319/sensors_19_02610.pdf.

Depuydt, Criel, De Letter and van Mierlo (2024), published ERP methods reproduced in Bio-protocol, section 2.3 ERP preprocessing, list a 150 uV maximum-difference criterion and a 0.5 uV low-activity criterion over 100 ms, alongside additional rejection rules. https://bio-protocol.org/exchange/minidetail?id=19622089&type=30. This supports amplitude and low-activity screening in ERP work. It does not establish the 0.1 uV cutoff used here.

The MNE Epochs API defines reject as a maximum per-channel epoch peak-to-peak threshold and flat as a minimum per-channel epoch peak-to-peak threshold. https://mne.tools/stable/generated/mne.Epochs.html. The 0.1 uV cutoff remains a project-defined near-flat safeguard. No verified source found in this search establishes 0.1 uV whole-epoch PTP as an optimal criterion for Nencki-Symfonia. It must be described as a chosen parameter, with its observed rejection counts reported.
