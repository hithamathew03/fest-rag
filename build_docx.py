"""Converts the report story (reportlab flowables built in build_report.py) into a native, editable Word file."""
import re, html
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_BREAK, WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from reportlab.platypus import Paragraph, Table, Image, PageBreak, Preformatted, KeepTogether, Spacer

NAVY = RGBColor(0x1F, 0x3B, 0x63)


def shade(cell, hexcol):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcol)
    tcPr.append(sh)


def add_runs(par, markup, size=None, base_font=None):
    """Very small parser for the reportlab mini-markup we use: <b>, <i>, <font face='Courier' size=..>, <br/>."""
    markup = markup.replace("<br/>", "\n")
    tokens = re.split(r"(<[^>]+>)", markup)
    bold = ital = mono = False
    for t in tokens:
        if not t:
            continue
        if t.startswith("<"):
            tag = t.lower()
            if tag == "<b>": bold = True
            elif tag == "</b>": bold = False
            elif tag == "<i>": ital = True
            elif tag == "</i>": ital = False
            elif tag.startswith("<font") and "courier" in tag: mono = True
            elif tag == "</font>": mono = False
            continue
        parts = html.unescape(t).split("\n")
        for k, piece in enumerate(parts):
            if k:
                par.add_run().add_break()
            if piece:
                r = par.add_run(piece); r.bold = bold or None; r.italic = ital or None
                if mono or base_font:
                    r.font.name = "Courier New" if mono else base_font
                if size:
                    r.font.size = Pt(size)


def style_of(fl):
    return getattr(fl.style, "name", "B")


def emit(doc, fl):
    if isinstance(fl, KeepTogether):
        for x in fl._content:
            emit(doc, x)
    elif isinstance(fl, PageBreak):
        doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    elif isinstance(fl, Spacer):
        doc.add_paragraph()
    elif isinstance(fl, Preformatted):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        for k, line in enumerate(fl.lines):
            if k:
                p.add_run().add_break()
            r = p.add_run(line); r.font.name = "Courier New"; r.font.size = Pt(7.5)
    elif isinstance(fl, Image):
        doc.add_picture(fl.filename, width=Cm(min(fl.drawWidth / 28.35, 17.5)))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif isinstance(fl, Paragraph):
        st = style_of(fl)
        if st in ("H1", "H2", "H3"):
            h = doc.add_heading(level={"H1": 1, "H2": 2, "H3": 3}[st])
            add_runs(h, fl.text)
        elif st in ("t", "t2", "t3", "t4"):
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, fl.text, size={"t": 20, "t2": 14, "t3": 13, "t4": 15}[st])
            for r in p.runs:
                r.font.color.rgb = NAVY
        else:
            p = doc.add_paragraph()
            size = {"SM": 8, "NOTE": 10, "ans": 9}.get(st)
            add_runs(p, fl.text, size=size)
            if st == "NOTE":
                pPr = p._p.get_or_add_pPr(); sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "FFF4D6"); pPr.append(sh)
            if st == "ans":
                pPr = p._p.get_or_add_pPr(); sh = OxmlElement("w:shd")
                sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), "EEF5EE"); pPr.append(sh)
    elif isinstance(fl, Table):
        rows = fl._cellvalues
        ncol = len(rows[0])
        widths = [w / 28.35 for w in fl._colWidths]
        scale = min(1.0, 17.8 / sum(widths))
        t = doc.add_table(rows=len(rows), cols=ncol)
        t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER; t.autofit = False
        tblPr = t._tbl.tblPr
        lay = OxmlElement("w:tblLayout"); lay.set(qn("w:type"), "fixed"); tblPr.append(lay)
        for j, col in enumerate(t.columns):
            col.width = Cm(widths[j] * scale)
        for gc, w in zip(t._tbl.tblGrid.findall(qn("w:gridCol")), widths):
            gc.set(qn("w:w"), str(int(w * scale * 567)))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                cell = t.cell(i, j)
                cell.width = Cm(widths[j] * scale)
                para = cell.paragraphs[0]
                para.paragraph_format.space_after = Pt(0)
                if isinstance(val, (list, tuple)):  # reportlab wraps cell contents in a list after layout
                    val = next((v for v in val if isinstance(v, Paragraph)), " ".join(map(str, val)))
                txt = val.text if isinstance(val, Paragraph) else str(val)
                fs = getattr(getattr(val, "style", None), "fontSize", 8) if isinstance(val, Paragraph) else 8
                add_runs(para, txt, size=max(6.5, min(fs, 9)))
                if i == 0:
                    shade(cell, "DFE7F2")
                    for r in para.runs:
                        r.bold = True
        # repeat header row on each page
        trPr = t.rows[0]._tr.get_or_add_trPr(); h = OxmlElement("w:tblHeader"); h.set(qn("w:val"), "true"); trPr.append(h)
        doc.add_paragraph().paragraph_format.space_after = Pt(2)


def story_to_docx(story, path):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(1.6); sec.top_margin = Cm(1.5); sec.bottom_margin = Cm(1.6)
    base = doc.styles["Normal"]; base.font.name = "Calibri"; base.font.size = Pt(10)
    base.paragraph_format.space_after = Pt(4)
    for lvl, size in ((1, 15), (2, 12.5), (3, 11)):
        hs = doc.styles[f"Heading {lvl}"]; hs.font.name = "Calibri"; hs.font.size = Pt(size); hs.font.color.rgb = NAVY
        hs.font.bold = True
        hs.paragraph_format.space_before = Pt(10 if lvl == 1 else 6); hs.paragraph_format.space_after = Pt(4)
        hs.paragraph_format.keep_with_next = True
        rpr = hs.element.get_or_add_rPr(); rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
            rf.set(qn(a), "Calibri")
    fp = sec.footer.paragraphs[0]; fp.text = "Group-12 | Assignment-2 Modern Database Systems | FestiVox"
    for r in fp.runs:
        r.font.size = Pt(8)
    for fl in story:
        emit(doc, fl)
    doc.save(str(path))
