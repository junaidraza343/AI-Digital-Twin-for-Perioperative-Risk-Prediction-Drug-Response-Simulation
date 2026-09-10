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

## [17] — placeholder author list, now resolved

**Currently in the thesis:** `Authors. (2025). Digital twins in healthcare: a
comprehensive review and future directions. Frontiers in Digital Health, 7, 1633539.`

**Correct (Crossref, DOI 10.3389/fdgth.2025.1633539):**
> Khoshfekr Rudsari, H., Tseng, B., Zhu, H., & Song, L. (2025). Digital twins in
> healthcare: a comprehensive review and future directions. *Frontiers in Digital
> Health*, 7, 1633539. https://doi.org/10.3389/fdgth.2025.1633539

Journal, volume and article number were right; only the authors were a placeholder.

## [19] — placeholder authors AND the wrong journal

**Currently in the thesis:** `Authors. (2025). Digital twin for the formal analysis
of a depth of anesthesia controller. Anesthesiology / IEEE Transactions on
Biomedical Engineering.` Body text in Section 1.2.6 calls it "A 2025 paper in
**Anesthesiology**".

**Correct (Crossref, DOI 10.1177/00375497241311617):**
> AbdElSalam, M., Bensalem, S., Delacourt, A., & He, W. (2025). Digital twin for
> the formal analysis of a depth of anesthesia controller. *SIMULATION*, 101(3),
> 341–360. https://doi.org/10.1177/00375497241311617

It appeared in **SIMULATION**, not *Anesthesiology* and not *IEEE TBME*. Both the
reference entry and the sentence in Section 1.2.6 need correcting.

## [18] — does NOT resolve as cited

**Currently in the thesis:** `Kovatchev, B., et al. (2023). Whole-body metabolic
digital twin for type-2 diabetes management. Nature Medicine / npj Digital Medicine.`

No such paper could be located. Crossref returns no whole-body metabolic digital
twin paper by Kovatchev; his indexed work in this area is on glycemic risk
algorithms (*Diabetes Technology & Therapeutics* 2003). The digital-twin-for-T2D
literature the sentence describes appears to be **Shamanna and colleagues**, e.g.

> Shamanna, P., Joshi, S., Thajudeen, M., & Shah, L. (2024). Personalized nutrition
> in type 2 diabetes remission: application of digital twin technology.
> *Frontiers in Endocrinology*, 15, 1485464. https://doi.org/10.3389/fendo.2024.1485464

**Do not simply swap the name** — check which study was actually intended, then
cite it properly. A reference naming two alternative journals with a slash is
itself a signal the source was never verified.

## [11] — completely wrong authors

**Currently in the thesis:** `Roh, S. Y., Lee, K. Y., & colleagues. (2025). Machine
learning methods for the prediction of intraoperative hypotension with biosignal
waveforms. Medicina, 61(11), 2039.`

**Correct (Crossref, DOI 10.3390/medicina61112039):**
> Shim, J.-G., Yoon, W., Lee, S. J., Chang, S.-H., Jung, S.-R., & Chung, J. Y.
> (2025). Machine learning methods for the prediction of intraoperative hypotension
> with biosignal waveforms. *Medicina*, 61(11), 2039.
> https://doi.org/10.3390/medicina61112039

Title, journal, volume, issue and article number are all correct — the author list
is of a different paper entirely. Section 1.2.4's "Roh and colleagues [11]" must
become "Shim and colleagues [11]".

## [13] — wrong first author, missing volume and pages

**Currently in the thesis:** `Wang, Y., et al. (2025). Towards reliable prediction
of intraoperative hypotension: a cross-center evaluation of deep learning-based and
MAP-derived methods. Journal of Clinical Monitoring and Computing.`

**Correct (Crossref, DOI 10.1007/s10877-025-01357-0):**
> Chaari, N., Winski, G., Hallbäck, M., Lundström, N., Björne, H., & Jacobsson, M.
> (2025). Towards reliable prediction of intraoperative hypotension: a cross-center
> evaluation of deep learning-based and MAP-derived methods. *Journal of Clinical
> Monitoring and Computing*, 40(1), 43–57.
> https://doi.org/10.1007/s10877-025-01357-0

Section 1.2.5's "A 2025 cross-center evaluation by Wang and colleagues [13]" must
become "by Chaari and colleagues [13]". This reference matters more than most — it
is the source of the selection-bias argument the whole project is built around, so
an examiner is likely to follow it.

## [10] — wrong authors, and it reveals a shuffle with [9]

**Currently in the thesis:** `Jo, Y. Y., Jang, J. H., Kwon, J., Lee, H. C., Jung,
C. W., Byun, S., & Jeong, H. G. (2021). Predicting intraoperative hypotension using
deep learning with waveforms of arterial blood pressure, electroencephalogram, and
electrocardiogram: STEP-OP. JMIR Medical Informatics, 9(9), e31311.`

**Correct (Crossref, DOI 10.2196/31311):**
> Choe, S., Park, E., Shin, W., Koo, B., & Shin, D. (2021). Short-term event
> prediction in the operating room (STEP-OP) of five-minute intraoperative
> hypotension using hybrid deep learning. *JMIR Medical Informatics*, 9(9), e31311.
> https://doi.org/10.2196/31311

Journal, volume, issue and article number are right; the authors and the title are
not.

### The two errors are connected

- **[9]** is attributed to **Choe** but is actually **Jeong** et al.
- **[10]** is attributed to **Jo** but is actually **Choe** et al.

"Choe" has migrated from entry [10] to entry [9]. This is the signature of author
lists being shifted by one during reference-manager import or manual renumbering,
not of two independent mistakes. **Check every entry either side of these**, and
check that the numbered citations in the body still point at the work each sentence
describes. Section 1.2.3's "Jo and colleagues [10] reported a STEP-OP system" should
read "Choe and colleagues [10]".

## Verified correct — no change needed

Checked against Crossref and found accurate in authors, journal, volume, issue and
pages:

| Ref | Work | DOI |
|---|---|---|
| [3] | Lee & Jung, Vital Recorder, *Sci Rep* 2018;8:1527 | 10.1038/s41598-018-20062-4 |
| [4] | Lee et al., VitalDB, *Sci Data* 2022;9:279 | 10.1038/s41597-022-01411-5 |
| [5] | Bijker et al., *Anesthesiology* 2007;107(2):213-220 | 10.1097/01.anes.0000270724.40897.8e |
| [6] | Hatib et al., *Anesthesiology* 2018;129(4):663-674 | 10.1097/aln.0000000000002300 |
| [7] | Wijnberge et al., HYPE, *JAMA* 2020;323(11):1052-1060 | 10.1001/jama.2020.0592 |
| [8] | Lee et al., *BJA* 2021;126(4):808-817 | 10.1016/j.bja.2020.12.035 |
| [15] | Corral-Acero et al., *Eur Heart J* 2020;41(48):4556-4564 | 10.1093/eurheartj/ehaa159 |

Every one of these spells its authors out in full — consistent with the pattern
below.

## Still unverified

Not yet DOI-checked: the PK-PD model sources (Eleveld 2018, Joachim 2024) and
refs [1], [2], [14], [16], [18], [20]-[25].

**Seven of the thesis's citations are wrong: [9], [10], [11], [12], [13], [17],
[19].** Two distinct failure modes:

1. **Abbreviated entries** — every reference written as "Authors.", "& colleagues"
   or a bare "et al." was misattributed. Entries that spell their authors out in
   full have so far all checked out.
2. **A shift between [9] and [10]** — "Choe" belongs to [10] but sits on [9]. That
   is a mechanical error, so neighbouring entries deserve a look even where they
   look plausible.

## Prior art that must be cited

DIGPHAT / Woillard et al., *Therapie* 2026;81(2):147–158,
DOI 10.1016/j.therap.2025.09.006 — publishes the general "digital pharmacological
twin" concept. The novelty claim must be scoped to the operating-room integration
rather than to the twin concept itself.
