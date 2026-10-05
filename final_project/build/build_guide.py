"""Presentation guide built from presentation_script.md."""
from html import escape
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate

R = Path(__file__).resolve().parent
md = (R / 'presentation_script.md').read_text()
(R / 'documents/Presentation_Guide.md').write_text(md)
H1 = ParagraphStyle('h1', fontName='Helvetica-Bold', fontSize=18, leading=23, spaceBefore=8, spaceAfter=10)
H2 = ParagraphStyle('h2', fontName='Helvetica-Bold', fontSize=12.5, leading=16, spaceBefore=6, spaceAfter=6)
B = ParagraphStyle('b', fontName='Helvetica', fontSize=10.2, leading=14.4, spaceAfter=8)
st = []
for blockt in [x.strip() for x in md.split('\n\n') if x.strip()]:
    for line in blockt.split('\n'):
        if line.startswith('## '):
            st.append(Paragraph(escape(line[3:]), H2))
        elif line.startswith('# '):
            st.append(Paragraph(escape(line[2:]), H1))
        elif line.strip():
            st.append(Paragraph(escape(line), B))
SimpleDocTemplate(str(R / 'documents/Presentation_Guide.pdf'), pagesize=A4, leftMargin=48, rightMargin=48,
                  topMargin=44, bottomMargin=48, title='Poster presentation guide',
                  author='Shreya Verma; Chih-yu Liu; William Sun').build(st)
print('guide built')
