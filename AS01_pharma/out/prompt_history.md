# Prompt history

Reference provenance: independently reviewed and finalized by Shreya Verma.

## v1 original

Full baseline prompt, used for all 25 records:

```text
You label documents from a corpus about pharmaceutical and therapeutic biotechnology job postings. Fill in every field:
- job_title: Copy job_title from the source table exactly. -- a short value taken from the document.
- employer: Copy employer from the source table exactly. -- a short value taken from the document.
- location: Copy the full location from the source table exactly; not_stated if absent. -- a short value taken from the document.
- job_function: Classify the primary responsibility. clinical_development includes clinical operations and clinical data; commercial_medical_affairs includes field medical and medical affairs; quality includes QC and QA; data_technology is computational/IT roles unless primarily clinical data. Use unclear if insufficient evidence. -- one of research_discovery, clinical_development, regulatory, quality, manufacturing_supply, commercial_medical_affairs, data_technology, corporate_other, unclear.
- work_arrangement: Use only an explicit statement about this specific role or an explicit remote/hybrid designation in its location. A city alone does not imply onsite. Travel is not remote work. Use not_stated for unresolved alternatives. -- one of onsite, hybrid, remote, not_stated.
- required_qualifications: Copy complete source sentences or bullet text stating required skills, knowledge, certifications or experience. Include education alternatives intact. A Qualifications/Requirements section is required unless explicitly preferred. Exclude responsibilities and generic culture/benefits. -- a list of strings.
- preferred_qualifications: Copy complete source sentences or bullet text explicitly preferred, desired, a plus or nice-to-have. Preserve wording; no paraphrase or inferred preferences. -- a list of strings.
- education_requirements: Copy complete source sentences or bullets specifying required education. Preserve all OR alternatives, equivalent-experience clauses and linked experience amounts together. Do not reduce a PhD/MS/BS alternative to a single degree. Exclude preferred-only education. -- a list of strings.
- experience_requirements: Copy complete source sentences or bullets specifying required numeric years of experience. Preserve ranges, plus signs and education-dependent alternatives. Exclude preferred-only years and company history. -- a list of strings.
```

## Observed failure and selection

All schema-valid baseline records with an incorrect work_arrangement; selected before running revised prompt. The recurring failure is inference of work mode without sufficient role-specific evidence or confusion with generic employer/compensation text. Invalid JSON records are excluded from this targeted diagnostic because they do not provide a schema-valid work-mode prediction. No revised results were available during selection.

- pharma_arrowheadpharmacareers_5188281007: baseline `onsite`; reference `not_stated`.
- pharma_arrowheadpharmacareers_5188291007: baseline `onsite`; reference `not_stated`.
- pharma_arrowheadpharmacareers_5213007007: baseline `onsite`; reference `not_stated`.
- pharma_arrowheadpharmacareers_5234169007: baseline `onsite`; reference `not_stated`.
- pharma_beamtherapeutics_8392998002: baseline `onsite`; reference `not_stated`.
- pharma_beamtherapeutics_8583823002: baseline `onsite`; reference `hybrid`.
- pharma_beamtherapeutics_8636699002: baseline `onsite`; reference `not_stated`.
- pharma_beamtherapeutics_8795551002: baseline `onsite`; reference `not_stated`.
- pharma_beamtherapeutics_8817599002: baseline `onsite`; reference `not_stated`.
- pharma_beamtherapeutics_8817994002: baseline `onsite`; reference `not_stated`.
- pharma_kymeratherapeutics_7726507003: baseline `onsite`; reference `not_stated`.
- pharma_kymeratherapeutics_7785662003: baseline `onsite`; reference `not_stated`.
- pharma_kymeratherapeutics_7991240003: baseline `onsite`; reference `not_stated`.
- pharma_relaytherapeutics_5985754004: baseline `onsite`; reference `not_stated`.
- pharma_relaytherapeutics_6123386004: baseline `onsite`; reference `not_stated`.
- pharma_revolutionmedicines_7695103003: baseline `onsite`; reference `hybrid`.
- pharma_revolutionmedicines_7720354003: baseline `onsite`; reference `hybrid`.
- pharma_revolutionmedicines_7836119003: baseline `onsite`; reference `remote`.
- pharma_revolutionmedicines_7984060003: baseline `onsite`; reference `not_stated`.

## v2 recovery

Motivation: force an evidence check and distinguish role tags from compensation boilerplate. Only work-arrangement instructions change; schema, reference labels, full input and decoding settings stay fixed. All original-prompt instructions are retained.

```text
You label documents from a corpus about pharmaceutical and therapeutic biotechnology job postings. Fill in every field:
- job_title: Copy job_title from the source table exactly. -- a short value taken from the document.
- employer: Copy employer from the source table exactly. -- a short value taken from the document.
- location: Copy the full location from the source table exactly; not_stated if absent. -- a short value taken from the document.
- job_function: Classify the primary responsibility. clinical_development includes clinical operations and clinical data; commercial_medical_affairs includes field medical and medical affairs; quality includes QC and QA; data_technology is computational/IT roles unless primarily clinical data. Use unclear if insufficient evidence. -- one of research_discovery, clinical_development, regulatory, quality, manufacturing_supply, commercial_medical_affairs, data_technology, corporate_other, unclear.
- work_arrangement: Use only an explicit statement about this specific role or an explicit remote/hybrid designation in its location. A city alone does not imply onsite. Travel is not remote work. Use not_stated for unresolved alternatives. -- one of onsite, hybrid, remote, not_stated.
- required_qualifications: Copy complete source sentences or bullet text stating required skills, knowledge, certifications or experience. Include education alternatives intact. A Qualifications/Requirements section is required unless explicitly preferred. Exclude responsibilities and generic culture/benefits. -- a list of strings.
- preferred_qualifications: Copy complete source sentences or bullet text explicitly preferred, desired, a plus or nice-to-have. Preserve wording; no paraphrase or inferred preferences. -- a list of strings.
- education_requirements: Copy complete source sentences or bullets specifying required education. Preserve all OR alternatives, equivalent-experience clauses and linked experience amounts together. Do not reduce a PhD/MS/BS alternative to a single degree. Exclude preferred-only education. -- a list of strings.
- experience_requirements: Copy complete source sentences or bullets specifying required numeric years of experience. Preserve ranges, plus signs and education-dependent alternatives. Exclude preferred-only years and company history. -- a list of strings.

WORK-ARRANGEMENT EVIDENCE PROCEDURE (apply only to work_arrangement):
1. Locate an explicit work-mode statement for this specific job in its description, source location, or role-specific recruitment tags. Interpret #LI-Remote as remote and #LI-Hybrid as hybrid. Explicit weekly partial onsite attendance (for example 1-3 days per week) is hybrid; explicit full-time onsite or five in-office days is onsite.
2. A city, company headquarters address, laboratory/manufacturing responsibilities, travel requirement, or the word Field does NOT establish onsite, hybrid, or remote work. Do not infer a mode from what a job normally requires.
3. Generic compensation text about salaries for candidates working onsite is conditional salary boilerplate, NOT this role's work arrangement. It must not override #LI-Remote, #LI-Hybrid, or an explicit role-specific statement.
4. If there is no explicit role-specific work-mode evidence, output not_stated. If explicit alternatives remain unresolved, output not_stated.
Examples: location Cambridge, MA with no role-specific mode statement -> not_stated. Location Redwood City with #LI-Remote plus generic onsite salary boilerplate -> remote.
Keep every other field and its existing extraction rules unchanged. Do not add evidence keys to the JSON schema.
```

## Outcome

Actual before/after outcomes, fixes and regressions are saved in recovery.json; raw output is saved in phi_v2.jsonl. The rerun diagnoses known errors and is not a held-out performance estimate. No additional prompt versions were tried.
