# Citation corrections (verified 2026-09-09)

Each entry below was checked against Crossref / the publisher, not against a
secondary source. Reference numbers refer to `FYP_Chapter1.docx`.

## [9] — wrong authors, wrong issue, wrong pages

**Currently in the thesis:**
> Choe, S. H., Cho, H., Bae, J., Ji, S.-H., Yoon, H.-K., Lee, H.-C., Lee, J.-H.,
> Kim, H.-S., & Kim, J.-T. (2024). Prediction of intraoperative hypotension using
> deep learning models based on non-invasive monitoring devices. *Journal of
> Clinical Monitoring and Computing*, 38(5), 1101–1110.

**Correct (Crossref, DOI 10.1007/s10877-024-01206-6):**
> Jeong, H., Kim, D., Kim, D. W., Baek, S., Lee, H.-C., Kim, Y., & Ahn, H. J.
> (2024). Prediction of intraoperative hypotension using deep learning models
> based on non-invasive monitoring devices. *Journal of Clinical Monitoring and
> Computing*, 38(6), 1357–1365. https://doi.org/10.1007/s10877-024-01206-6

The title is right; the author list, issue and page range are all wrong. Body text
in Section 1.2 that reads "Choe and colleagues [9]" must become "Jeong and
colleagues [9]". The reported figures (AUROC 0.917 / 0.833) match this paper.

## [12] — wrong first author; the VitalDB claim is CORRECT

**Currently in the thesis:**
> Liu, J., et al. (2026). Transformer-based deep learning model for real-time
> prediction of intraoperative hypotension using dynamic time-series vital signs:
> a retrospective study. *PLOS Medicine*, 23(3), e1005024.

**Correct (Crossref, DOI 10.1371/journal.pmed.1005024):**
> Zhu, S., Shi, W., Qian, H., Tong, X., Hu, R., Bo, J., & Gu, X. (2026).
> Transformer-based deep learning model for real-time prediction of intraoperative
> hypotension using dynamic time-series vital signs: A retrospective study.
> *PLOS Medicine*, 23(3), e1005024. https://doi.org/10.1371/journal.pmed.1005024

Journal, volume, issue, article number and year are correct. Only the author
attribution is wrong ("Liu, J." -> "Zhu, S.").

**Note — an earlier research pass claimed this study did NOT use VitalDB. That
claim was wrong.** The paper trains on 319,699 cases from Nanjing Drum Tower
Hospital and externally validates on **5,260 VitalDB cases**. The thesis sentence
"externally validated on VitalDB" is accurate and should be kept.

## Still unverified

These were flagged earlier and have NOT yet been DOI-checked: Hatib 2018,
Wijnberge 2020 (HYPE), Enevoldsen 2022, Lee 2021 BJA, Jo 2021, Eleveld 2018,
Joachim 2024, Vital Recorder 2018, VitalDB 2022, Corral-Acero 2020, and refs
[17]/[19] which are cited generically ("A 2025 comprehensive review", "A 2025
paper in Anesthesiology") and need concrete author attribution.

## Prior art that must be cited

DIGPHAT / Woillard et al., *Therapie* 2026;81(2):147–158,
DOI 10.1016/j.therap.2025.09.006 — publishes the general "digital pharmacological
twin" concept. The novelty claim must be scoped to the operating-room integration
rather than to the twin concept itself.
