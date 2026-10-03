"""Oct 3 revision of the team report content.

Changes: full author name (Chih-yu Li), expanded related work, cited data
sources, expanded discussion, complete reference list, appendix rows for the
as-submitted AS02 PDF and the data-source index. No experimental number is
changed; every number below was re-verified against the saved run files.
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parent
src = json.loads((R.parent / 'final_pass_20261002' / 'report_content.json').read_text())
p = json.loads(json.dumps(src))


def block(page, start):
    for b in p[page]['blocks']:
        if 'text' in b and b['text'].startswith(start):
            return b
    raise KeyError(start)


def text(t):
    return {'text': t}


def head(h):
    return {'heading': h}


# Page 0: authors, date, contributions
block(0, 'Shreya Verma')['text'] = (
    'Shreya Verma · Chih-yu Li · William Sun\n'
    '95820 Applications of NLX and LLM · Carnegie Mellon University\n3 October 2026')
block(0, 'The team contributions')['text'] = (
    'The team contributions supplied for this synthesis are Shreya Verma’s Pharma corpus and AS01/AS02 '
    'evidence [29], Chih-yu Li’s Finance corpus and memo [30], and William Sun’s Tech corpus and report [31]. '
    'The sector packages do not all contain the same experiment artifacts. Findings below distinguish locally '
    'audited files from numbers available only in a teammate’s memo.')

# Page 1: related work and data sources
rw = [
    text('Documentation of data and models. Datasheets for Datasets [1] and Data Statements [2] motivate '
         'documenting collection, composition and intended use. We apply that principle by retaining source '
         'URLs, record units and sampling limits instead of treating the combined corpus as a random '
         'labor-market sample. Model Cards [3] motivate reporting performance in the conditions relevant to an '
         'intended use; here schema validity, source agreement, refusals and human review are reported as '
         'separate outcomes.'),
    text('Small models and the runtime. Phi-4-mini is a 3.8-billion-parameter instruction-tuned model whose '
         'technical report and model card [4, 5] emphasize instruction following and function calling. We run a '
         'community 4-bit MLX conversion [6, 7] through the LLMBox modes and tool dispatcher [8]; this '
         'quantized local execution is a particular implementation, not a reproduction of the reported '
         'benchmarks.'),
    text('Structured output. Constrained decoding can guarantee that generated text matches a grammar or '
         'schema [10], whereas a format request in the prompt only asks for it. Tam et al. [11] report that '
         'format restrictions can also reduce answer quality. Our C1 condition requests JSON and validates it '
         'afterwards with Pydantic [9] without constrained decoding, which is why Markdown fences can cause a '
         'complete schema failure even when the enclosed object is usable.'),
    text('Tools, injection and refusal. Toolformer [12] shows models learning to call external tools; our tool '
         'is deliberately limited to one read-only lookup. Prompt injection [13] and indirect injection '
         'through retrieved content [14] motivate our planted-instruction probes with canary tokens, and the '
         'OWASP list [15] names both injection and sensitive-information disclosure as leading LLM-application '
         'risks. XSTest [16] shows that safety measures can refuse legitimate requests, so we report '
         'over-refusal next to the catch rate. For toxicity, RealToxicityPrompts [17] illustrates '
         'classifier-based measurement, Llama Guard [18] a model-based safeguard and Zheng et al. [19] the '
         'strengths and biases of LLM judges; we chose a transparent lexicon and human adjudication instead. '
         'The NIST AI Risk Management Framework [20] informs the measured-versus-precautionary policy marks.'),
]
blocks = p[1]['blocks']
i = next(k for k, b in enumerate(blocks) if b.get('text', '').startswith('Datasheets for Datasets'))
blocks[i:i + 1] = rw

block(1, 'Pharma contains complete')['text'] = (
    'Pharma contains complete therapeutic biotechnology and pharmaceutical postings from eight employers, '
    'retrieved from the public Greenhouse Job Board API [25] on September 20, 2026. All 200 records have text '
    'and tables. Finance uses an employer-industry definition: 18 Greenhouse boards [25], two Lever boards [26] '
    'and the City of New York’s NYC Jobs open dataset [27], retrieved September 19, 2026. Its 148 mixed and 52 '
    'text records include whole postings and some extracted sections. Tech focuses on early-career technology '
    'roles from public career pages on Greenhouse, Workday, Ashby, SmartRecruiters, iCIMS, Eightfold, Lever, '
    'BambooHR and employer sites [28], retrieved September 18, 2026, with 115 text, 134 table and one mixed '
    'record. Its company-overview, responsibility, qualification and compensation fragments can describe the '
    'same source posting.')
i = next(k for k, b in enumerate(blocks) if b.get('text', '').startswith('Finance’s supplied README'))
blocks.insert(i + 1, text(
    'Provenance granularity also differs. Pharma’s sources.csv lists all 200 posting URLs. Finance lists 21 '
    'board-level sources and Tech 72 board- or page-level sources; their record counts sum to 200 and 250, but '
    'individual posting URLs are not all listed row by row. Licensing notes state that public access is not an '
    'open redistribution license, so the combined corpus remains private coursework material.'))

# Page 2: methodology citations
b = block(2, 'The actual LLMBox Modes')
b['text'] = b['text'].replace('The actual LLMBox Modes methods and dispatcher run',
                              'The actual LLMBox [8] Modes methods and dispatcher run')
b['text'] = b['text'].replace('local 4-bit Phi-4-mini-instruct.', 'local 4-bit Phi-4-mini-instruct [6, 7].')
b = block(2, 'Primary outcomes are strict')
b['text'] = b['text'].replace('Primary outcomes are strict schema validity',
                              'Primary outcomes are strict Pydantic [9] schema validity')
b = block(2, 'The AS01 studies use different schemas')
b['text'] = b['text'].replace(
    'Pharma has 25 labels',
    'Pharma clusters were produced with TF-IDF features and k-means (k = 6) in scikit-learn [21]. Pharma has 25 labels')

b = block(2, 'The frozen 200-record Pharma corpus')
b['text'] = b['text'].replace('with seed 820.', 'with seed 820, following the course’s AS02 evaluation protocol [32].', 1)

# Page 3: larger-model citations
b = block(3, 'Pharma’s saved comparison')
b['text'] = (b['text']
             .replace('a mixture of Gemini model identifiers', 'a mixture of Gemini model identifiers [22]')
             .replace('Azure GPT-5-mini on 25 records', 'Azure GPT-5-mini [24] on 25 records')
             .replace('Groq-hosted gpt-oss-120b', 'Groq-hosted gpt-oss-120b [23]'))
b = block(4, 'C1’s zero strict schema')
b['text'] = b['text'].replace('C1’s zero strict schema passes were strongly affected by Markdown wrappers.',
                              'C1’s zero strict schema passes were strongly affected by Markdown wrappers, '
                              'consistent with requesting rather than constraining the format [10, 11].')

# Page 5: safety citations and expanded discussion
b = block(5, 'Thirty authored probes')
b['text'] = b['text'].replace('indirect injection and leakage (five each)',
                              'indirect injection [14] and leakage [15] (five each)')
b = block(5, 'No injection canary')
b['text'] =b['text'].replace('A five-term toxicity lexicon flagged',
                              'A five-term toxicity lexicon, chosen over a classifier [17, 18] or LLM judge [19] '
                              'for transparency, flagged')
b = block(5, 'Thirty authored probes')
b['text'] = b['text'].replace('It also blocks 1/10 benign requests.',
                              'It also blocks 1/10 benign requests, an over-refusal rate of the kind XSTest [16] '
                              'warns about and above the 5% ceiling in the initial policy.')
disc = [
    text('Who is affected. The benefits and risks fall on different people. Advisers may save preparation '
         'effort, while students bear the cost of incorrect advice: a wrong work arrangement wastes an '
         'application, and a truncated “or equivalent experience” clause can discourage a qualified '
         'applicant. Employers are affected when a summary misstates their posting, even though they never use '
         'the system.'),
    text('What the results mean for the design. The measurements locate the problem. Retrieval works (50/50 '
         'tool executions) and the source table already contains title, employer, location and URL, so a '
         'deterministic lookup is preferable for those fields. The model’s difficulty lies in output format '
         'and interpretation: strict validation rejected every C1 answer, and even the best condition matched '
         'all five scored fields on 17 of 50 postings. Format enforcement through constrained decoding [10] is '
         'therefore a better next investment than larger token budgets or higher temperature, which made B3 '
         'worse than B1.'),
    text('What the results mean for governance. A verifier that withholds unsupported answers (C4) reached '
         'complete safety by delivering nothing, and the keyword guardrail refused 10% of benign requests. '
         'Safety and service must be reported together. The measurements support keeping adviser review on '
         'every output and do not yet support student-facing use. Proposed controls include local processing '
         'of public postings, a selected-record read-only tool, exclusion of applicant data and adviser review '
         'before use. Authentication, retention enforcement and institutional approval are not implemented '
         'production controls.'),
    text('Across sectors. The Finance and Tech reports show the same pattern under different schemas: '
         'larger hosted models improved schema compliance, while list-valued fields such as skills or '
         'qualifications remained hard to match exactly. Because schemas, references and providers differ, '
         'this is a shared qualitative lesson rather than a measured ranking of models or sectors.'),
]
blocks = p[5]['blocks']
i = next(k for k, b in enumerate(blocks) if b.get('text', '').startswith('The benefits and risks fall'))
blocks[i:i + 1] = disc

# Page 6: future work citation
b = block(6, 'A first revision should')
b['text'] = b['text'].replace('A first revision should normalize complete JSON fences before validation',
                              'A first revision should test constrained decoding [10] or normalize complete JSON '
                              'fences before validation')

# Page 7: full references
refs = [
    'Literature and methods',
    '[1] Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., & Crawford, K. (2021). Datasheets for datasets. Communications of the ACM, 64(12), 86–92. https://arxiv.org/abs/1803.09010',
    '[2] Bender, E. M., & Friedman, B. (2018). Data statements for natural language processing: Toward mitigating system bias and enabling better science. Transactions of the ACL, 6, 587–604. https://aclanthology.org/Q18-1041',
    '[3] Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. Proceedings of FAT* 2019. https://arxiv.org/abs/1810.03993',
    '[4] Microsoft (2025). Phi-4-Mini technical report: Compact yet powerful multimodal language models via mixture-of-LoRAs. arXiv:2503.01743. https://arxiv.org/abs/2503.01743',
    '[5] Microsoft. Phi-4-mini-instruct model card. https://huggingface.co/microsoft/Phi-4-mini-instruct (accessed September 28, 2026).',
    '[6] Hannun, A., Digani, J., Katharopoulos, A., & Collobert, R. (2023). MLX: Efficient and flexible machine learning on Apple silicon; and mlx-lm. https://github.com/ml-explore/mlx',
    '[7] MLX Community. Phi-4-mini-instruct-4bit (quantized checkpoint used in all local runs). https://huggingface.co/mlx-community/Phi-4-mini-instruct-4bit',
    '[8] Kingsley, S. LLMBox. https://github.com/sarakingsley/llmbox ; experiment snapshot commit 23f97e77009cf5fe6f4d46b8a68234d18f9d2a30.',
    '[9] Colvin, S., et al. Pydantic (v2) data validation library. https://docs.pydantic.dev',
    '[10] Willard, B. T., & Louf, R. (2023). Efficient guided generation for large language models. arXiv:2307.09702. https://arxiv.org/abs/2307.09702',
    '[11] Tam, Z. R., Wu, C.-K., Tsai, Y.-L., Lin, C.-Y., Lee, H.-y., & Chen, Y.-N. (2024). Let me speak freely? A study on the impact of format restrictions on performance of large language models. Proceedings of EMNLP 2024: Industry Track. https://arxiv.org/abs/2408.02442',
    '[12] Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. NeurIPS 2023. https://arxiv.org/abs/2302.04761',
    '[13] Perez, F., & Ribeiro, I. (2022). Ignore previous prompt: Attack techniques for language models. NeurIPS ML Safety Workshop. https://arxiv.org/abs/2211.09527',
    '[14] Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. AISec ’23. https://arxiv.org/abs/2302.12173',
    '[15] OWASP Foundation (2025). OWASP Top 10 for Large Language Model Applications. https://genai.owasp.org/llm-top-10/',
    '[16] Röttger, P., Kirk, H., Vidgen, B., Attanasio, G., Bianchi, F., & Hovy, D. (2024). XSTest: A test suite for identifying exaggerated safety behaviours in large language models. NAACL 2024, 5377–5400. https://aclanthology.org/2024.naacl-long.301',
    '[17] Gehman, S., Gururangan, S., Sap, M., Choi, Y., & Smith, N. A. (2020). RealToxicityPrompts: Evaluating neural toxic degeneration in language models. Findings of EMNLP 2020. https://arxiv.org/abs/2009.11462',
    '[18] Inan, H., et al. (2023). Llama Guard: LLM-based input-output safeguard for human-AI conversations. arXiv:2312.06674. https://arxiv.org/abs/2312.06674',
    '[19] Zheng, L., et al. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS 2023 Datasets and Benchmarks. https://arxiv.org/abs/2306.05685',
    '[20] National Institute of Standards and Technology (2023). Artificial Intelligence Risk Management Framework (AI RMF 1.0), NIST AI 100-1. https://doi.org/10.6028/NIST.AI.100-1',
    '[21] Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.',
    'Comparison models',
    '[22] Google. Gemini API models documentation (gemini-flash-latest and gemini-3.5-flash, as recorded in the Pharma AS01 comparison). https://ai.google.dev/gemini-api/docs/models',
    '[23] OpenAI (2025). gpt-oss-120b & gpt-oss-20b model card. arXiv:2508.10925; served through Groq, https://console.groq.com/docs/models (Finance comparison).',
    '[24] OpenAI (2025). GPT-5 system card. https://openai.com/index/gpt-5-system-card/ ; GPT-5-mini accessed through Azure OpenAI (Tech comparison).',
    'Data sources',
    '[25] Greenhouse Job Board API (public, unauthenticated GET). https://developers.greenhouse.io/job-board.html. Pharma boards (retrieved 2026-09-20): arrowheadpharmacareers, beamtherapeutics, kymeratherapeutics, recursionpharmaceuticals, relaytherapeutics, revolutionmedicines, tesseratherapeutics, ultragenyxpharmaceutical. Finance boards (retrieved 2026-09-19): Jane Street, Carta, Jump Trading, Brex, Point72, Virtu Financial, Stripe, Gemini, IMC Trading, Mercury, AQR Capital Management, Affirm, Robinhood, SoFi, Betterment, Coinbase, Chime, Upstart.',
    '[26] Lever Postings API. https://github.com/lever/postings-api. Finance boards: Anchorage Digital, Alloy (retrieved 2026-09-19).',
    '[27] City of New York. NYC Jobs (dataset kpav-sd4t), NYC Open Data, used under the NYC Open Data Terms of Use. https://data.cityofnewyork.us/City-Government/Jobs-NYC-Postings/kpav-sd4t',
    '[28] Public employer career pages for the Tech corpus (retrieved 2026-09-18): 15 Greenhouse, 15 Workday, 11 Ashby, 3 SmartRecruiters, 3 iCIMS, 2 Eightfold, 2 Lever, 1 BambooHR and 20 direct employer sites; every source and its terms note is listed in sector_sources/tech/sources.csv.',
    'Team artifacts and course materials',
    '[29] Verma, S. (2026). Pharma AS01 corpus, labels, predictions and metrics; AS02 LLMBox-integrated evaluation and prototype timing records. Supplied project artifacts.',
    '[30] Li, C.-y. (2026). Finance AS01 corpus, reference labels, README and memo. assignment1_chihyul3.zip. Reported numerical results are not independently reproduced here.',
    '[31] Sun, W. (2026). Tech AS01 corpus, labels, README and report. 95820-assignment1-yiqings2.zip. Reported numerical results are not independently reproduced here.',
    '[32] Carnegie Mellon University, 95820 Applications of NLX and LLM (Fall 2026). Assignment 1, Assignment 2 and Final Project instructions.',
]
blocks = p[7]['blocks']
i = next(k for k, b in enumerate(blocks) if 'heading' in b)
new = []
for r in refs:
    new.append(head(r) if not r.startswith('[') else text(r))
blocks[:i] = new
tab = next(b for b in blocks if 'table' in b)['table']
tab.insert(-2, ['Data source index', 'sector_sources/*/sources.csv (650 records; per-source terms notes); references [25]–[28]'])
tab.insert(-2, ['AS02 report as submitted', 'documents/AS02_Report_as_submitted.pdf (the version uploaded to Canvas)'])

(R / 'report_content.json').write_text(json.dumps(p, ensure_ascii=False, indent=1))
print('pages', len(p), 'refs', sum(1 for r in refs if r.startswith('[')))

p[0]['title'] = 'Job Posting Extraction Across Pharma, Finance and Tech'
(R / 'report_content.json').write_text(json.dumps(p, ensure_ascii=False, indent=1))
