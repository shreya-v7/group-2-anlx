# Poster presentation guide

Team: Shreya Verma, Chih-yu Li, William Sun

## Three minute overview

Shreya, approximately 60 seconds: “Our project asks whether a small language model can help career advisers summarize job postings accurately. We combined Pharma, Finance and Tech material. The first surprise was the unit of data: 650 records represent 446 distinct source URLs. Some records are whole postings and others are short sections. We therefore cannot interpret row counts as hiring demand or pool the three studies into one accuracy score.”

Chih-yu, approximately 60 seconds: “The sector studies expose different measurement problems. In Finance, certification agreement looked high, but the memo reports that all four positive credential cases were missed. Most reference lists were empty. Finance reference labels also require independent review. In Tech, prompt recovery improved some outputs on a selected failed subset, which is useful development evidence but not a fresh test. These distinctions matter when a stakeholder asks whether a model is accurate.”

William, approximately 60 seconds: “The reproducible Pharma API experiment adds the operational picture. Its tool ran on all 50 requests, yet only 17 final answers matched five basic source fields. A structured-output condition failed strict JSON validation, often because of Markdown fences. Removing the fences changes the diagnosis, but requires a new test before adoption. Our recommendation is to retrieve existing metadata directly and keep any generated interpretation under adviser review.”

## Thirty second version

“We studied job posting extraction across Pharma, Finance and Tech. Our 650 records describe 446 source URLs, so records and jobs are not interchangeable. In the Pharma API, all 50 tool calls worked but only 17 final answers matched five source fields. Formatting, semantic accuracy and safety are different tests. We recommend direct source lookup for existing facts and human review for generated interpretation.”

## Role rotation

Suggested first round: Shreya presents while Chih-yu and William visit peer posters. Second round: Chih-yu presents. Third round: William presents. Adjust to the session schedule so every member presents and gives feedback. These are proposed roles, not a record of participation.

# Demonstration and questions

## A short visitor activity

Ask the visitor: “If a model returns the correct facts inside a Markdown code block, should a strict JSON API accept the answer?” Then show the poster’s C1 row and explain that an integration failure differs from a factual failure. Reveal the post-hoc 43/50 schema result, followed by the stricter 31/50 source-and-quote result. Ask what further review is needed before advising a student. Use saved artifacts rather than generating a new answer during the presentation.

## Why not report one team accuracy

The schemas, record units, annotation provenance and model conditions differ. A pooled percentage would hide those differences. The team contribution is the synthesis of failure modes and the reproducible Pharma API case, not a controlled contest between industries.

## Does zero attack success prove safety

No. The probes were few and authored for this task; a real secret was not present. Human output judgments apply only to unchanged responses. The guardrail also refused one benign request, so usefulness must be reported alongside blocking.

## Can we claim twelve postings to break even

Only as the earlier prototype’s conditional scenario: 90 seconds of review plus 60% times 135 seconds of correction equals 171 seconds. Compared with 330 seconds manual, that saves 159 seconds; 1,800 seconds of assumed setup divided by 159 rounds up to twelve. It is not an organization-wide estimate and does not validate the new API.

## Was the policy written before the experiments

For the final series, yes: V1 was frozen before fresh inference and V2 followed the results. Earlier experiments had already informed V1. The rerun preserves that history and does not create a blind holdout.

## Why not deploy the best condition

Seventeen of fifty exact source-field matches is inadequate for unattended advice; even perfect source strings would not guarantee complete qualifications. The team recommends a new shared evaluation with independent review and measured correction effort.

# Conference participation worksheet

Complete this during the actual session. The provided AS03 copy lists the feedback and nomination survey links as forthcoming; use the links and categories published by the teaching team.

## Peer project one

Project and team: ______________________________________________
Specific strength observed: _______________________________________
Evidence or evaluation question: __________________________________
Actionable suggestion and reason: _________________________________
Verbal feedback given by / time: __________________________________
Survey submitted by / time: _______________________________________

## Peer project two

Project and team: ______________________________________________
Specific strength observed: _______________________________________
Evidence or evaluation question: __________________________________
Actionable suggestion and reason: _________________________________
Verbal feedback given by / time: __________________________________
Survey submitted by / time: _______________________________________

## Nominations

For each official competition category, record an eligible peer project, the concrete evidence supporting the nomination and the survey confirmation. Do not nominate this team’s own project.
Category / project / reason: _______________________________________
Category / project / reason: _______________________________________
Category / project / reason: _______________________________________
Add rows for all categories supplied in the actual survey.

## Submission and attendance

One team member uploads the report PDF, poster PDF, supporting evidence archive and code ZIP as permitted by Canvas. All members attend and rotate roles. The supplied assignment lists October 9, 2026 at 11:59 pm, with no late submissions. Check Canvas for the session details, current survey links and upload receipt. No upload, presentation, feedback or nomination has been completed by preparing these files.