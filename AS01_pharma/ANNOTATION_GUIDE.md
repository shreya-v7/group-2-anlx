# Human-label guide - Shreya / Pharma

**Human annotation: DONE (25/25).** Shreya independently reviewed and finalized every field for the fixed evaluation sample. The records use `annotation_origin: human`, `annotator: Shreya Verma`, and `human_reviewed: true`.

The 25-record sample is fixed with Python seed 820. `out/annotation_evidence.json` preserves the source-line audit trail, and `label_review.html` provides the complete source text alongside every label.

## Consistent decisions

| Field | Labeling rule |
|---|---|
| job_title / employer / location | Copy the matching source table value. Preserve complete location strings and alternatives. Use `not_stated` for an absent location. |
| job_function | Choose the primary responsibility, not the employer's business or the reporting department alone. Use the enum descriptions in `taxonomy.json`. |
| work_arrangement | Use an explicit role-specific onsite, hybrid, or remote statement. A city alone does not establish onsite work; required travel is not remote work. Unresolved choices = `not_stated`. |
| required_qualifications | Full sentences or bullets about required skills, knowledge, experience, education, or certifications. Exclude duties, benefits, culture, and recruiting boilerplate. |
| preferred_qualifications | Full sentences or bullets explicitly introduced as preferred, desired, a plus, nice-to-have, bonus, or ideally. |
| education_requirements | Full required education sentence/bullet, preserving OR alternatives and experience substitutions. May also occur in required_qualifications. |
| experience_requirements | Full required numeric-experience sentence/bullet, preserving ranges and degree-dependent conditions. May also occur in required_qualifications. |

Qualifications, Requirements, Your Background, and Your Experience sections default to required unless a preference cue overrides them. A preference heading governs its bullets until the next heading. A clear role-specific requirement outside these sections also counts. Do not label company-history dates, general industry descriptions, or accommodation policies as qualifications.

For a mixed bullet with independently stated required and preferred clauses, copy the whole bullet into both qualification lists so the original logical relationship remains visible. Do not invent a minimum degree or years value from alternatives. Use complete source sentences/bullets as the comparison unit, not arbitrary keyword fragments. Lists are compared case-insensitively as sets; order and duplicate items do not matter. Internal punctuation and wording still matter. If no item is stated, return an empty list after reviewing the full source. Never use `null` in extraction labels.

Examples below are invented solely to explain labeling, not corpus records:

- “BS with 5+ years or MS with 3+ years” stays intact in required_qualifications, education_requirements and experience_requirements.
- “HPLC experience preferred” belongs in preferred_qualifications, not required_qualifications.
- “Location: Boston, MA” without a work-mode statement yields work_arrangement=`not_stated`.

## Recovery experiment

Read `out/evaluation.json` after baseline inference. Find a repeated error; record at least two affected document IDs and the mistaken field. Then set `prompts.v2_recovery` in `taxonomy.json` to a complete revised system prompt that retains the field definitions and adds the targeted rule. Keep `v1_original` unchanged. Save every prompt version and its motivation in `out/prompt_history.md`.

Run `python extraction.py --stage recovery --model-path /path/to/phi-4-mini-instruct`. This reruns baseline failures only. Report before/after on that same subset; it is a diagnostic recovery experiment, not an independent held-out estimate. If the revised prompt causes regressions in previously correct fields, report those too. If no failures occur, report that no recurring failure was observed; do not fabricate one.

## Privacy review

The collected records contain public employer advertisements, not applicant records. The provided regex screen found shared corporate contact emails and no phone-like spans; it is not a comprehensive entity audit. Pharmaceutical disease/drug mentions are not identifiable patient records by themselves. If you find personal names, personal contact details, patient data, private financial identifiers, or confidential text, use the provided Phi masking script and review its output before final submission. Rebuild the sample if corpus text changes.

## Decisions fixed before Phi inference

Analytical CMC development serving manufacturing/process readiness is `manufacturing_supply`; discovery biology is `research_discovery`. Clinical specimen coordination and clinical data management are `clinical_development`. Corporate compliance and finance are `corporate_other`. Commercial regional operations are `commercial_medical_affairs`. These boundaries are interpretive and should be discussed when evaluating disagreements.

Role-specific `#LI-Remote` and `#LI-Hybrid` tags override generic onsite salary boilerplate. One to three expected onsite days per week is treated as hybrid. `Field` alone is not an explicit remote designation. The misspelled `18+ yeas` is preserved verbatim as numeric experience; spelled-out one year counts. Generic work-authorization, culture, and recruitment notices are excluded. Mixed required/preferred bullets appear in both lists. Source duplicates are deduplicated as exact strings.
