# Poster presentation guide

Team: Shreya Verma, Chih-yu Liu, William Sun. About four minutes in total, then questions.

## Shreya: opening (about 75 seconds)

[Point at the title and "The question"] “Hi, I’m Shreya, and with Chih-yu and William we asked one question: can a small AI model help a university career office summarize job postings? Advisers pull the same facts out of every posting: title, employer, location, remote or onsite, and requirements. We wanted to know whether an AI could write that first draft for an adviser to check. Never to make hiring decisions.”

[Point at the data table] “Each of us built a dataset for one industry: I took Pharma, Chih-yu Finance and William Tech. That is 650 records from 446 source pages, all public job postings. One early lesson: Tech’s 250 rows come from only 71 postings, because many rows are pieces of one posting. More rows does not mean more jobs.”

[Point at "How the API works"] “For Pharma I built an API around Microsoft’s Phi-4-mini, running entirely on a laptop so no data leaves the machine. A posting goes in, the model extracts the facts, automated checks compare them with the source, and an adviser reviews the result. I wrote the usage rules before running the final tests, then tested six designs on 50 postings each. Chih-yu will walk you through what we found.”

## Chih-yu Liu: findings (about 75 seconds)

[Point at the Finding 1 chart] “Thanks, Shreya. The headline: the tool works, but the answers don’t. In the best design the model used its lookup tool correctly 50 out of 50 times, yet only 17 of 50 answers got all five basic facts right. A working tool does not guarantee a correct answer.”

[Point at Finding 2] “The most surprising result: when we forced a strict data format, the model scored zero out of 50. The facts were not the problem. It wrapped every answer in extra formatting that the checker rejected. Shreya fixed the parser and tested it on 50 new postings with the success rule written down first: 44 answers were valid and 33 passed every fact check. How you measure can change the conclusion completely.”

[Point at the three-sector table] “My Finance study showed the same trap from another angle. Certification extraction scored 86% agreement, but most postings list no certification, and on the four that did, the model missed every one. A high score can hide failure on exactly the cases that matter. Across all three industries, larger hosted models fixed the format, but lists of skills and qualifications stayed hard. William will cover his API study, safety and what we recommend.”

## William Sun: second API, safety, recommendation (about 90 seconds)

[Point at Finding 5] “Thanks, Chih-yu. I built a second API on the same model for Tech postings, with different code and data. Two design choices mattered. My parser removed the formatting before checking, and when an answer failed, my verifier repaired it against the posting instead of withholding it. That kept 39 of 50 answers usable, 34 by a stricter count. Shreya’s strict verifier would have passed 31 of 50 with the same formatting fix, so the two studies agree: format is an engineering problem, and the hard part is still reading. None of my 25 hand-checked records was fully right.”

[Point at Finding 3] “We also attacked our own APIs. In Pharma, the guardrail blocked 15 of 20 attacks but also 1 in 10 normal requests, above our 5% limit. In Tech, with 70 probes, I found something worse: the guardrail checks the model’s answer, not the posting. Five of six postings with discriminatory requirements, like coded age or religion criteria, went straight through as normal jobs.”

[Point at Finding 4 and Recommendation] “On cost, a posting took about 5.5 minutes by hand and about 3 with the AI plus review, but that saving disappears if review takes longer than about 4 minutes. So our recommendation: copy facts that already exist directly, use the AI only with an adviser checking every answer, screen the postings themselves for discriminatory requirements, and never use it to rank applicants. Happy to take questions.”

## Who answers which question

Why a small model? Shreya: privacy and cost; postings never leave the laptop.

Why not just use a big model? Chih-yu: it formats better, but sending data to a hosted service needs a separate privacy approval.

Is it safe? William: no attack succeeded on Pharma’s 30 probes, and 2 of 40 succeeded with the Tech guardrail, but probe sets we wrote ourselves cannot prove general security, and discriminatory postings are a gap.

Biggest limitation? Shreya: small samples (200 Pharma postings, 25 hand labels per sector) from one snapshot in time, and previously inspected evaluation data.

## A short visitor activity

Ask the visitor: “If the AI gets every fact right but in the wrong format, should the system accept it?” Point to the 0/50 strict result, the 43/50 result after removing the formatting, and William’s 39/50 with repair. Then ask what review is still needed before advising a student.

## Role rotation

Suggested: Shreya presents first while Chih-yu and William visit other posters; then Chih-yu; then William. Every member should present and give feedback, so everyone should know all three parts. These are proposed roles, not a record of participation.
