"""Render the editable memo.md. No evaluation numbers are invented or inferred."""
from pathlib import Path
import html, re
import reportlab
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

ROOT=Path(__file__).resolve().parent
INK=colors.HexColor('#15283E'); TEAL=colors.HexColor('#006D73')

def markup(text):
    text=html.escape(text)
    text=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',text)
    text=re.sub(r'`(.*?)`',r'<font name="Courier">\1</font>',text)
    return text

def render():
    font_dir=Path(reportlab.__file__).parent/'fonts'
    pdfmetrics.registerFont(TTFont('MemoSans',str(font_dir/'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('MemoSans-Bold',str(font_dir/'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('MemoSans',normal='MemoSans',bold='MemoSans-Bold',italic='MemoSans',boldItalic='MemoSans-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='MemoTitle',fontName='MemoSans-Bold',fontSize=17.5,leading=22,textColor=INK,spaceAfter=10))
    styles.add(ParagraphStyle(name='MemoHeading',fontName='MemoSans-Bold',fontSize=11,leading=14,textColor=TEAL,spaceBefore=10,spaceAfter=5))
    styles.add(ParagraphStyle(name='MemoBody',fontName='MemoSans',fontSize=8.6,leading=11.6,textColor=INK,spaceAfter=7))
    styles.add(ParagraphStyle(name='MemoCell',fontName='MemoSans',fontSize=8.3,leading=11,textColor=INK))
    body=[]; lines=(ROOT/'memo.md').read_text().splitlines();i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line: i+=1;continue
        if line=='---PAGE---':body.append(PageBreak());i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[s.strip() for s in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',c) for c in cells):rows.append(cells)
                i+=1
            widths=[205,295] if len(rows[0])==2 else [345,65,90]
            table=Table([[Paragraph(markup(c),styles['MemoCell']) for c in row] for row in rows],colWidths=widths,repeatRows=1)
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#DCEDEE')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#F3F6F8'),colors.white]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),0.6,TEAL)]))
            body.extend([table,Spacer(1,8)]);continue
        if line.startswith('# '):body.append(Paragraph(markup(line[2:]),styles['MemoTitle']))
        elif line.startswith('## '):body.append(Paragraph(markup(line[3:]),styles['MemoHeading']))
        else:body.append(Paragraph(markup(line),styles['MemoBody']))
        i+=1
    def footer(canvas,doc):
        canvas.setStrokeColor(TEAL);canvas.line(48,39,548,39)
        canvas.setFont('MemoSans',7);canvas.setFillColor(INK)
        canvas.drawString(48,26,'SHREYA  /  PHARMA JOB CORPUS  /  RESULTS MEMO')
        canvas.drawRightString(548,26,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'memo.pdf'),pagesize=(596,842),leftMargin=48,rightMargin=48,topMargin=42,bottomMargin=52,title='Pharma Job Market - Shreya',author='Shreya Verma')
    doc.build(body,onFirstPage=footer,onLaterPages=footer)
    print('Wrote memo.pdf from memo.md')

if __name__=='__main__':render()
