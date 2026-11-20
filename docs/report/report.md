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

## Preprocessing

### Input data and event alignment

Preprocessing covered all 42 Oddball recordings, with 127 recorded EEG channels per participant and an original sampling frequency of 1,000 Hz. Standard (S 5) and target (S 6) events defined 24,440 candidate binary trials. Distractor and response markers remained in the original recordings. Stimulus onsets from the event tables were checked against the BrainVision annotations before epoching.

### Processing steps and rationale

1. Channel checks identified previously marked bad channels, non-finite values, and channels that were constant across the recording. Usable EEG channels were re-referenced to their common average.
2. Continuous EEG was filtered with a fourth-order Butterworth band-pass filter at 0.5–30 Hz, applied forward and backward. This reduced slow drift and high-frequency activity outside the selected analysis band.
3. ICA was fitted independently for each participant using up to 30 components and a fixed random seed of 42. Fitting used a 250 Hz copy with an additional 1 Hz high-pass filter. The saved decomposition was applied to the continuous 0.5–30 Hz data. A 1 Hz high-pass copy for fitting is supported by the MNE ICA guidance. [5]
4. For sub-01, component 0 was selected after reviewing its time course alongside Fp1 and Fp2. For the other participants, automatic selection required a component to be an MNE candidate for both frontal channels and to have an absolute correlation of at least 0.9 with each. This frontal-proxy rule is a project heuristic because dedicated EOG channels were unavailable. Its threshold is a parameter requiring quality inspection. Applying the rule removed components in 28 further participants. For 13 participants, no component met the rule. We retained their filtered EEG without component removal as the decision for this processing version, avoiding removal on weaker frontal-proxy evidence. Their original automated review-required status remains in the quality logs for traceability.
5. Following reconstruction, the selected channels were F3, Fz, F4, T7, C3, Cz, C4, T8, CP3, CP4, P3, Pz, P4, PO7, PO8 and Oz. ICA and the common-average reference used the available full recording before this selection. This 16-electrode configuration has precedent in P300 classification research. [6]
6. Signals and event sample positions were jointly resampled to 128 Hz. Epochs covered approximately -200 to 1,000 ms on the resampled time grid. The prestimulus interval up to 0 ms provided the baseline. Baseline-corrected samples from 0 to 1,000 ms were retained for the model, giving 129 time points including both endpoints.
7. An epoch was rejected when any selected EEG channel had peak-to-peak amplitude above 150 µV or below 0.1 µV across the full epoch. Peak-to-peak amplitude is the maximum minus the minimum within a channel and epoch. MNE implements both maximum-amplitude and minimum-amplitude rejection this way. [7]

A published ERP study used a 150 µV peak-to-peak criterion in overlapping 200 ms windows stepped by 100 ms. Our whole-epoch criterion is an adaptation with a different observation window. The 0.1 µV cutoff was chosen as a near-flat safeguard. Its numerical value has not been validated here as optimal for this dataset. Both values should be interpreted alongside rejection counts and signal inspection. [7,8]

### Reasons for the selected processing strategy

The initial loss of 392 out of 579 candidate sub-01 trials prompted inspection of channel amplitudes and ICA components. Widespread frontal transients motivated ocular-artifact correction before amplitude rejection. The sub-01 correction recovered trials while keeping the rejection thresholds fixed.

The literature-based 16-channel configuration reduces input size while retaining spatial coverage relevant to the Oddball ERP analysis. Selecting channels after ICA preserves the use of frontal channels as ocular-artifact proxies during cleaning. Sub-01 retained the same number of trials with 16 channels as with 127 after ICA correction. Classification performance with the selected channels remains to be evaluated.

All 42 participants were retained under the selected quality criteria. Behavioral errors reported for sub-02, sub-14 and sub-17 were considered separately from stimulus-based labels. For the 13 participants without components meeting the automatic threshold, retaining the filtered EEG avoids forcing uncertain component removals. Residual artifacts are an acknowledged limitation of this version.

### Investigation of excessive trial rejection

The first sub-01 run tested the amplitude thresholds across 127 channels before removing ICA components. It retained 175/498 standards and 12/81 targets. Fp2 exceeded 150 µV in 391 of 579 candidate epochs, while the diagnostic table showed no channel below the 0.1 µV threshold. Fp1 and other frontal channels were also frequently flagged.

The component-0 trace showed repeated transients synchronized with large bilateral Fp1/Fp2 peaks. After its reviewed exclusion, 496/498 standards and 81/81 targets were retained at the same thresholds. Selecting 16 channels after this ICA correction retained the same 577 epochs. The pilot therefore motivated artifact correction before rejection.

![Sub-01 EEG before and after reviewed component removal.](figures/sub-01-before-after.png)

Figure 1. Sub-01 EEG before and after the reviewed removal of component 0. The separate pilot component time-course inspection motivated this decision. The overlay records its effect on frontal and midline channels.

![Sub-02 EEG before and after component removal.](figures/sub-02-before-after.png)

Figure 2. Sub-02 EEG before and after removal of component 0. Large frontal transients decreased. Over the displayed 30–90 second interval, Fp1 peak-to-peak amplitude decreased from approximately 486.5 to 98.8 µV and Fp2 from 472.1 to 92.0 µV. Cz and Pz had similar interval peak-to-peak amplitudes before and after processing. The overlay illustrates a local signal-quality check; model performance will be evaluated separately.

### Cohort outputs and remaining quality issues

The final epoch-generation run completed for all 42 participants with no recorded errors. Each saved array contains 16 channels and 129 time points per retained trial. Signal values are stored in volts, with channel names, relative sample times, participant identity, original stimulus sample positions, and labels (0 standard, 1 target).

| Class | Candidate trials | Retained trials | Rejected trials |
|---|---:|---:|---:|
| Standard | 21,018 | 20,827 | 191 |
| Target | 3,422 | 3,399 | 23 |
| Total | 24,440 | 24,226 | 214 |

Overall, 99.12% of candidate trials were retained. This represents the current processing version. ICA component removal was applied in 29 participants, including the manually reviewed sub-01. The other 13 were retained without component removal because no component met the automatic criterion. These participants are sub-03, sub-08, sub-12, sub-14, sub-18, sub-23, sub-25, sub-26, sub-30, sub-34, sub-35, sub-40 and sub-41. High retention alone does not establish artifact-free EEG.

| Participant | Retained standard | Retained target | ICA action |
|---|---:|---:|---|
| sub-01 | 496/498 | 81/81 | Reviewed removal of component 0 |
| sub-02 | 495/498 | 80/81 | Automatic removal of component 0 |
| sub-07 | 432/498 | 75/81 | Automatic removal of component 0 |
| sub-14 | 479/498 | 78/81 | No component met the automatic rule |
| sub-17 | 488/498 | 79/81 | Automatic removal of component 0 |

Sub-07 had the largest loss of trials and remains a priority for signal inspection. Sub-02, sub-14 and sub-17 were retained in the current dataset. Their known behavioral issues and the signal-quality checks address different aspects of the experiment. Counts for sub-23 and sub-38 begin above those of most participants; the present inventory alone does not explain the extra events.

The numerical results in this section were checked against the server-generated after_ica summary and participant split inventory. These counts define the preprocessing version selected for the current experiments. Possible residual ocular artifacts in the 13 participants without component removal remain a limitation.

## Participant-level Train/Test Preparation

A fixed shuffle with seed 42 assigned 34 participants to training and 8 to testing. The sets contain no shared participants and together cover all 42 participants. Channel order, time grid and per-participant class counts were validated by the split-generation script and checked against the preprocessing summary.

| Set | Participants | Standard trials | Target trials | Total trials |
|---|---:|---:|---:|---:|
| Train | 34 | 16,931 | 2,761 | 19,692 |
| Test | 8 | 3,896 | 638 | 4,534 |

The test participants are sub-02, sub-07, sub-08, sub-09, sub-15, sub-16, sub-18 and sub-41. Validation is currently empty. Model selection and early stopping will use participant-level validation within the training participants when required. Test results will be used for the final evaluation. The processing quality statuses remain attached to the participant reports.

## Classical Machine Learning

### Feature extraction and initial models

For each of 16 channels, the baseline-corrected 0–1000 ms signal was averaged over 20 consecutive 50 ms windows, producing 320 features per trial in channel-major order. The final window included the sample at 1000 ms. Timestamps defined window membership because 50 ms does not correspond to an integer number of samples at 128 Hz. Amplitudes were converted to microvolts.

We evaluated a majority-class baseline, LDA with automatic covariance shrinkage and equal class priors, and an RBF SVM with C=1, gamma=scale and balanced class weights. Standardization for LDA and SVM was fitted on training participants only. Initial parameters were fixed before test evaluation. All models used 19,692 training trials from 34 participants and 4,534 test trials from the same eight held-out participants.

### Initial held-out results

| Model | Macro-F1 | Standard correctly classified | Standard predicted target | Target predicted standard | Target correctly classified |
|---|---:|---:|---:|---:|---:|
| Majority baseline | 0.4622 | 3,896 | 0 | 638 | 0 |
| Shrinkage LDA | 0.7103 | 3,258 | 638 | 176 | 462 |
| RBF SVM | 0.7578 | 3,690 | 206 | 294 | 344 |

Confusion matrices use true labels as rows and predicted labels as columns, with standard followed by target. SVM achieved the highest macro-F1 of the initial models. LDA detected 462 of 638 targets (72.4%), while SVM detected 344 (53.9%). SVM generated fewer false target predictions, 206 compared with 638 for LDA. These results show different error tradeoffs. The majority baseline missed every target, illustrating the importance of class-sensitive evaluation.

The comparison concerns these initial configurations and this participant split. Subsequent hyperparameter selection should use participant-level validation inside the training set. The elapsed times recorded in the supplied JSON files include fitting and prediction and depend on the execution environment.

## References

1. Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. The Nencki-Symfonia electroencephalography/event-related potential dataset. *GigaScience*. 2022. [doi:10.1093/gigascience/giac015](https://doi.org/10.1093/gigascience/giac015).
2. Dzianok P, Antonova I, Wojciechowski J, Dreszer J, Kublik E. Nencki–Symfonia EEG/ERP dataset. [OpenNeuro ds004621](https://openneuro.org/datasets/ds004621). [GigaDB dataset DOI](https://doi.org/10.5524/100990).
3. OpenNeuroDatasets. [Oddball acquisition metadata](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/main/task-oddball_eeg.json) and [event dictionary](https://raw.githubusercontent.com/OpenNeuroDatasets/ds004621/main/task-oddball_events.json).
4. Project data acquisition inventory. [Oddball EEG dataset and acquisition](data_acquisition.pdf), p. 2.

5. MNE. [Repairing artifacts with ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html).
6. [Electrode subset selection methods for an EEG-based P300 brain-computer interface](https://pmc.ncbi.nlm.nih.gov/articles/PMC4133302/).
7. MNE. [Epochs API and peak-to-peak rejection](https://mne.tools/stable/generated/mne.Epochs.html).
8. Navid MS, Niazi IK, Lelic D, Drewes AM, Haavik H. (2019). The Effects of Filter’s Class, Cutoff Frequencies, and Independent Component Analysis on the Amplitude of Somatosensory Evoked Potentials Recorded from Healthy Volunteers. Sensors 19(11), 2610. [doi:10.3390/s19112610](https://doi.org/10.3390/s19112610).
