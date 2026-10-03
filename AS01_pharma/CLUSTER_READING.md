# Cluster reading — measured corpus analysis

TF-IDF word unigrams/bigrams, normalized truncated SVD (up to 128 dimensions), k-means with k=6, random_state=0 and n_init=10. Silhouette: 0.1709. The starter plot shows a two-dimensional PCA projection; visual distances do not preserve all relationships. Full descriptions, including employer boilerplate, were clustered.

Cluster names below are descriptive interpretations based on three seeded representatives per cluster and the complete employer cross-tabulation. They are interpretive names rather than validated occupational class labels.

## Cluster 0 — Kymera: protein degradation and development

**Records:** 29 · **Evaluation-sample records:** 3

All 29 records come from Kymera. The shared company introduction and values vocabulary (including TPD and degrader) recur before substantially different responsibilities. The sampled titles span value/access, program management, and translational medicine operations. This is primarily an employer/template grouping, not one occupational function.

**Representative records inspected:**

- `pharma_kymeratherapeutics_7864663003` — Senior Director, Outcomes, Value & Access ([source](https://www.kymeratx.com/careers-culture/working-at-kymera/?gh_jid=7864663003#open-positions)).
- `pharma_kymeratherapeutics_7894000003` — Director or Senior Director, Program Management ([source](https://www.kymeratx.com/careers-culture/working-at-kymera/?gh_jid=7894000003#open-positions)).
- `pharma_kymeratherapeutics_7965329003` — Senior Manager or Associate Director, Translational Medicine Operations ([source](https://www.kymeratx.com/careers-culture/working-at-kymera/?gh_jid=7965329003#open-positions)).

**Highest mean TF-IDF terms:** kymera, clinical, degrader, experience, tpd, regulatory, study, ll, development, improve.

## Cluster 1 — Relay: drug discovery and development operations

**Records:** 11 · **Evaluation-sample records:** 2

All 11 records come from Relay. The examples span scientific leadership, clinical vendor/contract management, and portfolio operations. Shared drug-discovery descriptions help explain the cluster even though the responsibilities differ. Do not interpret the drug/discovery top terms as proof that every member is a laboratory role.

**Representative records inspected:**

- `pharma_relaytherapeutics_6123386004` — VP, Head of Drug Discovery ([source](https://job-boards.greenhouse.io/relaytherapeutics/jobs/6123386004)).
- `pharma_relaytherapeutics_6129110004` — Senior Manager, Clinical Business Operations ([source](https://job-boards.greenhouse.io/relaytherapeutics/jobs/6129110004)).
- `pharma_relaytherapeutics_6129122004` — Senior Manager/Associate Director, Portfolio & Strategy Operations ([source](https://job-boards.greenhouse.io/relaytherapeutics/jobs/6129122004)).

**Highest mean TF-IDF terms:** drug, relay, discovery, motion, protein, clinical, development, cmc, process, built.

## Cluster 2 — Beam: gene editing and operational roles

**Records:** 36 · **Evaluation-sample records:** 6

All 36 records come from Beam. The repeated base-editing company overview appears in QC leadership and in both alliance-management and supply-chain co-op descriptions. The company platform vocabulary is common; seniority and function are heterogeneous.

**Representative records inspected:**

- `pharma_beamtherapeutics_8760769002` — Associate Director, Quality Control ([source](https://job-boards.greenhouse.io/beamtherapeutics/jobs/8760769002)).
- `pharma_beamtherapeutics_8795480002` — Alliance Management Co-op ([source](https://job-boards.greenhouse.io/beamtherapeutics/jobs/8795480002)).
- `pharma_beamtherapeutics_8795629002` — Supply Chain SAP Co-op ([source](https://job-boards.greenhouse.io/beamtherapeutics/jobs/8795629002)).

**Highest mean TF-IDF terms:** beam, editing, base, gene, overview, cell, experience, vision, op, manufacturing.

## Cluster 3 — Smaller employers: mixed therapeutic-biotech roles

**Records:** 19 · **Evaluation-sample records:** 1

This cluster combines 10 Recursion, six Tessera, and three Ultragenyx records. Examples include clinical supply-chain management, corporate legal transactions, and oligonucleotide chemistry. Read it as a heterogeneous remainder rather than a validated AI or clinical specialty. Only one randomly sampled evaluation record falls here, so any eventual accuracy estimate will be especially unstable.

**Representative records inspected:**

- `pharma_recursionpharmaceuticals_8214932` — Clinical Supply Chain Manager ([source](https://job-boards.greenhouse.io/recursionpharmaceuticals/jobs/8214932)).
- `pharma_tesseratherapeutics_5170380007` — Vice President, Corporate Transactions ([source](https://job-boards.greenhouse.io/tesseratherapeutics/jobs/5170380007)).
- `pharma_tesseratherapeutics_5240104007` — Senior Research Associate, Oligonucleotide Chemistry ([source](https://job-boards.greenhouse.io/tesseratherapeutics/jobs/5240104007)).

**Highest mean TF-IDF terms:** recursion, tessera, talent, ai, clinical, discovery, ultragenyx, experience, biology, development.

## Cluster 4 — Revolution Medicines: oncology and corporate roles

**Records:** 52 · **Evaluation-sample records:** 6

All 52 records come from Revolution Medicines. A long recurring oncology introduction, contact/privacy text, and compensation wording unite otherwise different QA, administrative and corporate-development responsibilities. Template structure could influence extraction difficulty, but no Phi run has established that relationship.

**Representative records inspected:**

- `pharma_revolutionmedicines_7836119003` — Senior QA Manager, GMP GxP Auditing & Vendor Management ([source](https://www.revmed.com/careers-list/?gh_jid=7836119003)).
- `pharma_revolutionmedicines_7867247003` — Senior Administrative Assistant II, Commercialization ([source](https://www.revmed.com/careers-list/?gh_jid=7867247003)).
- `pharma_revolutionmedicines_7984060003` — Senior Vice President, Head of Corporate Development ([source](https://www.revmed.com/careers-list/?gh_jid=7984060003)).

**Highest mean TF-IDF terms:** revolution, ras, revmed, medicines, email, experience, salary, pay, base, medical.

## Cluster 5 — Arrowhead: RNAi, quality and manufacturing context

**Records:** 53 · **Evaluation-sample records:** 7

All 53 records come from Arrowhead. The shared RNAi introduction explains prominent company/science vocabulary. QC, supplier quality and analytical-method validation appear in the inspected representatives; other functions also occur in this employer cluster. It should not be equated with a comprehensive manufacturing category.

**Representative records inspected:**

- `pharma_arrowheadpharmacareers_5206021007` — Associate Scientist III, Quality Control (2nd shift) ([source](https://arrowheadpharma.com/careers/?gh_jid=5206021007)).
- `pharma_arrowheadpharmacareers_5214722007` — Senior QA Specialist - Supplier Quality ([source](https://arrowheadpharma.com/careers/?gh_jid=5214722007)).
- `pharma_arrowheadpharmacareers_5230678007` — Senior Manager, Analytical Method Validation ([source](https://arrowheadpharma.com/careers/?gh_jid=5230678007)).

**Highest mean TF-IDF terms:** arrowhead, rnai, rna, wi, analytical, interference, liver, genes, silencing, manufacturing.

## Connection to reference vs. Phi

Human labeling is DONE (25/25). The following measured results use the actual local Phi baseline, before recovery. Full-record agreement requires all nine fields and a valid schema.

| Cluster | Sample n | Full-record matches | Field agreement | Most frequent field errors |
|---|---:|---:|---:|---|
| 0 | 3 | 0 | 44.4% (12/27) | job_function (3/3), work_arrangement (3/3), preferred_qualifications (3/3) |
| 1 | 2 | 0 | 50.0% (9/18) | work_arrangement (2/2), required_qualifications (2/2), preferred_qualifications (2/2) |
| 2 | 6 | 0 | 48.1% (26/54) | work_arrangement (6/6), job_function (5/6), required_qualifications (5/6) |
| 3 | 1 | 0 | 66.7% (6/9) | required_qualifications (1/1), preferred_qualifications (1/1), education_requirements (1/1) |
| 4 | 6 | 0 | 63.0% (34/54) | required_qualifications (5/6), work_arrangement (4/6), experience_requirements (4/6) |
| 5 | 7 | 0 | 33.3% (21/63) | employer (7/7), job_function (6/7), work_arrangement (6/7) |

Kymera records share a short qualifications section, while the sampled Arrowhead records include long responsibility/qualification lists and two malformed model responses. These observations help explain the measured field differences, but do not establish a causal effect of template length. Employer, writing style and occupation are confounded. The smaller-employer cluster has only one evaluation record, so its percentage is a case observation rather than a reliable ranking.

Across the corpus, titles and locations each match on 23/25 records, while work arrangement matches on 4/25 and required-qualification sets on 5/25. These results motivate testing work-mode evidence rules and caution against treating fluently structured outputs as complete extractions. Qualification paraphrases and punctuation changes are penalized by the exact-match metric.

A future sensitivity analysis could remove employer boilerplate only from clustering inputs and compare occupational coherence, retaining the original extraction inputs. That additional experiment was not run.
