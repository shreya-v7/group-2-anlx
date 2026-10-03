# Assignment 1: Corpus Construction and Characterization

## Group Topic

**Job Market** -- job postings and career data across different industries.

## Individual Subtopic

**Tech Jobs** -- entry-level and new-graduate software engineering, data science, ML, and infrastructure roles at technology companies and tech-adjacent organizations.

## Corpus Contents

The corpus contains 250 structured records extracted from 71 unique job postings across 68 companies. Each job posting was split into multiple records by document type (company overview, responsibilities, qualifications, compensation/benefits) to preserve distinct information units and increase the proportion of tabular records.

Record types include:
- Company and role overviews (narrative descriptions)
- Responsibility listings
- Qualification requirements (structured as requirement tables)
- Compensation and benefits (structured as component tables)

## Sources

Data was collected from public job posting pages hosted on:
- Ashby (jobs.ashbyhq.com) -- 50 records
- Greenhouse (job-boards.greenhouse.io, boards.greenhouse.io) -- 39 records
- Workday portals (NVIDIA, Visa, RELX, RTX, etc.) -- 25 records
- Amazon Jobs (amazon.jobs) -- 7 records
- Lever (jobs.lever.co) -- 6 records
- Other career pages -- remaining records

All sources are publicly accessible job postings. Licensing details per source are documented in `sources.csv`.

## Record Counts and Types

| Modality  | Count   | Percentage |
| --------- | ------- | ---------- |
| table     | 134     | 53.6%      |
| text      | 115     | 46.0%      |
| mixed     | 1       | 0.4%       |
| **Total** | **250** |            |

The 54% tabular share exceeds the 20% minimum requirement.

## Collection and Preprocessing Decisions

- **Record splitting:** One job posting produces 2-4 records. This approach preserves distinct information types (qualifications vs. compensation vs. overview) and naturally generates tabular records from structured sections.
- **Tabular extraction:** Qualification requirements were structured as objects with `category` (required/preferred) and `requirement` fields. Compensation data was structured with `component`, `value`, and `currency` fields.
- **Sensitive information:** No PII, PHI, FIN, or CONF masking was needed. Job postings are public-facing documents that do not contain sensitive personal information.
- **Doc ID renumbering:** Records were renumbered sequentially (tech_001 through tech_250) to resolve ID collisions from batch collection.

## Known Limitations

- Covers only new-grad/entry-level positions; no mid-career, senior, or executive roles.
- Single retrieval date (2026-09-18); no temporal dimension for trend analysis.
- Skewed toward US-based, English-language postings from a limited set of job boards.
- Many records are short fragments (median 79 characters) due to the splitting strategy.
- No application outcome or hiring pipeline data.
