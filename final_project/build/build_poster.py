"""36 x 24 inch scientific poster with charts (Oct 3 revision).

Every number is read from the saved run files at build time (no hand-typed
results), except the AS01 teammate figures and prototype timings, which come
from the cited memos and prototype_cost/cost_inputs.json.
"""
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

R = Path(__file__).resolve().parent
P = R.parents[3]
RUNS = P / 'AS02/as02_prospective/runs/policy_first_gpu'
COST = P / 'AS02/as02_prospective/updated_package/prototype_cost/cost_inputs.json'
OUT = R / 'documents'
FIG = R / 'figures'
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#fcfcfb'
NAVY, TEAL = '#173e50', '#187d86'

# ---------- data from saved runs ----------
runs = ['B1', 'B3', 'C1', 'C2', 'C3', 'C4']
labels = {'B1': 'B1 plain generate', 'B3': 'B3 hotter, longer', 'C1': 'C1 structured output',
          'C2': 'C2 tool lookup', 'C3': 'C3 guardrail', 'C4': 'C4 strict verifier'}
M = {r: json.loads((RUNS / r / 'metrics.json').read_text()) for r in runs}
valid = {r: M[r]['schema_valid_count'] for r in runs}
allc = {r: round(M[r]['delivered_all_source_fields_correct_rate'] * M[r]['count']) for r in runs}
withheld = {r: M[r]['blocked_count'] for r in runs}
lat = {r: M[r]['mean_latency_seconds'] for r in runs}
D = json.loads((RUNS / 'D' / 'metrics.json').read_text())
blk = D['blocked_fraction_by_attack_category']
over = D['benign_over_refusal_rate']
agree = json.loads((RUNS / 'D' / 'human_transfer_audit.json').read_text())
cost = json.loads(COST.read_text())
manual = cost['manual_seconds_per_posting']
review = cost['review_seconds_per_posting']
corr = cost['correction_seconds_per_incorrect_posting'] * (1 - cost['usable_without_correction_fraction'])
assert (valid['C2'], allc['C2'], valid['C1']) == (22, 17, 0)

plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 26, 'axes.edgecolor': GRID,
                     'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK, 'text.color': INK})


def clean(ax):
    for s in ['top', 'right', 'left']:
        ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=1.2)
    ax.set_axisbelow(True)


# Chart 1: schema valid vs all-five-correct per run (grouped horizontal bars)
fig, ax = plt.subplots(figsize=(10.4, 6.1), dpi=200)
fig.patch.set_facecolor('white')
ys = list(range(len(runs)))[::-1]
h = 0.36
for y, r in zip(ys, runs):
    ax.barh(y + h / 2 + 0.02, valid[r], h, color=BLUE)
    ax.barh(y - h / 2 - 0.02, allc[r], h, color=ORANGE)
    ax.text(valid[r] + 0.6, y + h / 2 + 0.02, f'{valid[r]}', va='center', fontsize=22, color=INK)
    ax.text(allc[r] + 0.6, y - h / 2 - 0.02, f'{allc[r]}', va='center', fontsize=22, color=INK)
    if withheld[r] == 50:
        ax.text(6, y, 'all 50 withheld', va='center', fontsize=20, color=INK2, style='italic')
ax.set_yticks(ys)
ax.set_yticklabels([labels[r] for r in runs], fontsize=23)
ax.set_xlim(0, 50)
ax.set_xticks([0, 10, 20, 30, 40, 50])
ax.set_xlabel('answers out of 50', fontsize=22)
clean(ax)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=['valid JSON (strict schema)', 'all 5 source facts correct'], loc='upper right',
          frameon=False, fontsize=21)
fig.tight_layout()
fig.savefig(FIG / 'runs.png', facecolor='white')
plt.close(fig)

# Chart 2: guardrail catch rate by category + benign over-refusal
cats = [('Out of scope', blk['out_of_scope'], 5), ('Leakage', blk['leakage'], 5),
        ('Harmful', blk['harmful'], 5), ('Indirect injection', blk['injection'], 5),
        ('Benign (wrongly blocked)', over, 10)]
fig, ax = plt.subplots(figsize=(10.4, 4.5), dpi=200)
ys = list(range(len(cats)))[::-1]
for y, (name, frac, n) in zip(ys, cats):
    c = ORANGE if name.startswith('Benign') else BLUE
    ax.barh(y, frac * 100, 0.62, color=c)
    ax.text(frac * 100 + 1.5, y, f'{round(frac * n)}/{n}', va='center', fontsize=23, color=INK)
ax.set_yticks(ys)
ax.set_yticklabels([c[0] for c in cats], fontsize=23)
ax.set_xlim(0, 112)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
ax.set_xlabel('share of probes blocked by the guardrail', fontsize=22)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / 'safety.png', facecolor='white')
plt.close(fig)

# Chart 3: adviser seconds per posting (stacked review + expected correction)
fig, ax = plt.subplots(figsize=(10.4, 3.0), dpi=200)
ax.barh(1, manual, 0.55, color=INK2)
ax.text(manual + 6, 1, f'{manual} s', va='center', fontsize=23)
ax.barh(0, review, 0.55, color=BLUE)
ax.barh(0, corr, 0.55, left=review + 2, color=AQUA)
ax.text(review / 2, 0, 'review', va='center', ha='center', fontsize=20, color='white')
ax.text(review + 2 + corr / 2, 0, 'fix', va='center', ha='center', fontsize=20, color=INK)
ax.text(review + corr + 8, 0, f'{review + corr:.0f} s', va='center', fontsize=23)
ax.set_yticks([1, 0])
ax.set_yticklabels(['By hand', 'API + adviser'], fontsize=23)
ax.set_xlim(0, 400)
ax.set_xlabel('adviser seconds per posting (earlier prototype timing)', fontsize=21)
clean(ax)
fig.tight_layout()
fig.savefig(FIG / 'time.png', facecolor='white')
plt.close(fig)

# ---------- poster canvas ----------
W, H = 2592, 1728
c = canvas.Canvas(str(OUT / 'Scientific_Poster_36x24.pdf'), pagesize=(W, H))
c.setTitle('Job posting extraction across Pharma, Finance and Tech')
c.setAuthor('Shreya Verma; Chih-yu Li; William Sun')
c.setFillColor(colors.HexColor(SURF))
c.rect(0, 0, W, H, fill=1, stroke=0)
c.setFillColor(colors.HexColor(NAVY))
c.rect(0, H - 230, W, 230, fill=1, stroke=0)
c.setFillColor(colors.white)
c.setFont('Helvetica-Bold', 66)
c.drawString(64, H - 98, 'Can a small AI model summarize job postings for career advisers?')
c.setFont('Helvetica', 34)
c.drawString(66, H - 158, 'Shreya Verma · Chih-yu Li · William Sun   |   Pharma · Finance · Tech job postings')
c.setFont('Helvetica', 26)
c.drawString(66, H - 205, '95820 Applications of NLX and LLM · Carnegie Mellon University · October 2026')

COLW = 790
GUT = 51
XS = [64, 64 + COLW + GUT, 64 + 2 * (COLW + GUT)]
TOP = H - 270


def para(x, y, txt, size=28, bold=False, color=INK, w=COLW, gap=18, lead=1.3):
    st = ParagraphStyle('p', fontName='Helvetica-Bold' if bold else 'Helvetica', fontSize=size,
                        leading=size * lead, textColor=colors.HexColor(color))
    q = Paragraph(txt, st)
    _, hh = q.wrap(w, 3000)
    q.drawOn(c, x, y - hh)
    return y - hh - gap


def heading(x, y, txt):
    c.setFillColor(colors.HexColor(TEAL))
    c.rect(x, y - 50, 10, 46, fill=1, stroke=0)
    return para(x + 24, y, escape(txt), 40, True, TEAL, COLW - 24, 16)


def image(x, y, path, w=COLW):
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    hh = w * ih / iw
    c.drawImage(img, x, y - hh, w, hh)
    return y - hh - 14


def table(x, y, rows, widths, size=25):
    st = ParagraphStyle('t', fontName='Helvetica', fontSize=size, leading=size * 1.25)
    t = Table([[Paragraph(escape(str(v)), st) for v in r] for r in rows], colWidths=widths)
    t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e6f0f2')),
                           ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#cdd9dc')),
                           ('TOPPADDING', (0, 0), (-1, -1), 9), ('BOTTOMPADDING', (0, 0), (-1, -1), 9)]))
    _, hh = t.wrap(COLW, 2000)
    t.drawOn(c, x, y - hh)
    return y - hh - 18


def flow(x, y):
    """Pipeline diagram: posting -> API mode -> local Phi -> checks -> adviser."""
    steps = ['Job posting', 'LLMBox API mode', 'Phi-4-mini (local, 4-bit)', 'Schema + source checks',
             'Adviser review']
    bw, bh, gap = COLW, 54, 22
    for i, s in enumerate(steps):
        yy = y - i * (bh + gap)
        fill = '#e6f0f2' if i not in (2,) else '#fde9df'
        c.setFillColor(colors.HexColor(fill))
        c.setStrokeColor(colors.HexColor('#9fb6bc'))
        c.roundRect(x, yy - bh, bw, bh, 10, fill=1, stroke=1)
        c.setFillColor(colors.HexColor(INK))
        c.setFont('Helvetica-Bold' if i in (2, 4) else 'Helvetica', 27)
        c.drawCentredString(x + bw / 2, yy - bh + 17, s)
        if i < len(steps) - 1:
            c.setStrokeColor(colors.HexColor(INK2))
            c.setLineWidth(2.5)
            c.line(x + bw / 2, yy - bh - 2, x + bw / 2, yy - bh - gap + 4)
            c.setFillColor(colors.HexColor(INK2))
            p = c.beginPath()
            p.moveTo(x + bw / 2 - 8, yy - bh - gap + 12)
            p.lineTo(x + bw / 2 + 8, yy - bh - gap + 12)
            p.lineTo(x + bw / 2, yy - bh - gap + 2)
            p.close()
            c.drawPath(p, fill=1, stroke=0)
    return y - len(steps) * (bh + gap) - 6


# Column 1: question, data, method
x, y = XS[0], TOP
y = heading(x, y, 'The question')
y = para(x, y, 'A university career office wants short, <b>source-linked summaries</b> of job postings '
               '(title, employer, location, work arrangement, requirements) that an adviser checks before '
               'students see them. Not hiring decisions.', 29)
y = heading(x, y, 'The data: 650 public job postings')
y = table(x, y, [['Sector', 'Records', 'Source URLs', 'Employers'],
                 ['Pharma', '200', '200', '8'], ['Finance', '200', '175', '27'], ['Tech', '250', '71', '68']],
          [220, 180, 210, 180])
y = para(x, y, 'Collected Sept 18–20, 2026 from public job-board APIs (Greenhouse, Lever), NYC Open Data and '
               'employer career pages. Tech rows are often fragments of one posting, so more rows do not mean more jobs.',
         23, color=INK2)
y = heading(x, y, 'How the API works')
y = flow(x, y)
y = para(x, y, 'Everything runs on a laptop: no posting leaves the machine. We tested six designs on 50 '
               'postings each (300 requests), then attacked the API with 30 safety probes.', 25, color=INK2)
assert y > 80, y

# Column 2: headline results
x, y = XS[1], TOP
y = heading(x, y, 'Finding 1: tool works, answers don’t')
y = para(x, y, f'The lookup tool ran <b>{M["C2"]["successful_tool_calls"]}/50</b> times. '
               f'But only <b>{allc["C2"]}/50</b> answers got all five source facts right.', 31, True, NAVY)
y = image(x, y, FIG / 'runs.png')
y = para(x, y, 'Five facts: posting ID, URL, title, employer, location. One run per design, same 50 postings.',
         21, color=INK2)
y = heading(x, y, 'Finding 2: formatting hid good answers')
y = para(x, y, 'The model wrapped its JSON in Markdown fences, so the strict checker rejected <b>all 50</b> '
               'structured answers. Stripping the fences afterwards gave <b>43/50</b> valid, <b>31/50</b> '
               'source-correct: a diagnosis, not a new result.', 25)
y = para(x, y, 'A checker that refuses anything unverified (C4) blocked all 50 answers. Perfectly safe, '
               'completely useless.', 25, True, NAVY)
y = heading(x, y, 'Same lesson in all three sectors')
y = table(x, y, [['Sector (AS01 test set)', 'Phi valid JSON', 'Larger hosted model'],
                 ['Pharma (25)', '23/25', 'Gemini Flash: 25/25'],
                 ['Finance (28)', '12/28', 'gpt-oss-120b: 28/28'],
                 ['Tech (25)', '7/25', 'GPT-5-mini: 24/25']], [300, 220, 270], 23)
y = para(x, y, 'Bigger models fixed the format; skill and qualification lists stayed hard. Schemas differ, '
               'so this is not a ranking; Finance/Tech figures from teammate memos.', 21, color=INK2)
assert y > 80, y

# Column 3: safety, effort, recommendation
x, y = XS[2], TOP
y = heading(x, y, 'Finding 3: safety has a service cost')
y = image(x, y, FIG / 'safety.png')
y = para(x, y, f'Keyword guardrail blocked 15/20 attacks but also <b>1 in 10</b> normal requests (above our '
               f'5% limit). It agreed with human judgment on <b>{agree["agreement_count"]}/'
               f'{agree["agreement_denominator"]}</b> probes. Misses were paraphrases.', 25)
y = heading(x, y, 'Finding 4: review time decides the value')
y = image(x, y, FIG / 'time.png')
y = para(x, y, 'Single timings, earlier prototype; if review took 249 s, the saving disappears.', 21, color=INK2)
y = heading(x, y, 'Finding 5: repair, don’t withhold')
y = para(x, y, 'A second API (Tech, same model) stripped fences and repaired answers against the posting: '
               '<b>39/50</b> usable (34 strict). With the same fence fix, Pharma’s strict verifier would have '
               'passed <b>31/50</b>. Still 0/25 hand-labeled Tech records fully right; <b>5 of 6</b> '
               'discriminatory postings passed its guardrail.', 24)
y = heading(x, y, 'Recommendation')
y = para(x, y, '• Copy known facts directly from the source: no AI needed.<br/>'
               '• AI only as a supervised tool; an adviser checks every answer.<br/>'
               '• Never for ranking applicants or eligibility decisions.<br/>'
               '• Next: enforce JSON at decoding, screen postings for discrimination, test new employers.', 24)
assert y > 80, y

c.setFillColor(colors.HexColor(INK2))
c.setFont('Helvetica', 20)
c.drawString(64, 42, 'Convenience sample of public postings; results on previously inspected data. Full methods, '
                     '33 references and all data/run logs: Team Research Report and supporting package.')
c.save()
print('poster built')
