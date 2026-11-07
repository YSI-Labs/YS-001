"""Convierte PREPRINT.md en PDF (A4) con reportlab: titulo, secciones con marcadores, tablas, figuras con leyenda, listas, codigo en linea. Uso: python -I build_pdf.py"""
import os, re, sys
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, ListFlowable, ListItem, CondPageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping

HERE = os.path.dirname(os.path.abspath(__file__)); FD = "C:/Windows/Fonts/"
for n, f in (("Body", "segoeui.ttf"), ("Body-B", "segoeuib.ttf"), ("Body-I", "segoeuii.ttf"), ("Code", "consola.ttf")):
    pdfmetrics.registerFont(TTFont(n, FD + f))
pdfmetrics.registerFont(TTFont("Body-BI", FD + "segoeuib.ttf"))
addMapping("Body", 0, 0, "Body"); addMapping("Body", 1, 0, "Body-B"); addMapping("Body", 0, 1, "Body-I"); addMapping("Body", 1, 1, "Body-BI")
INK = colors.HexColor("#1a1a1a"); ACC = colors.HexColor("#0b4f8a"); GREY = colors.HexColor("#555555"); LINE = colors.HexColor("#b9c2cc"); HEAD = colors.HexColor("#e8eef5")
S = dict(
    title=ParagraphStyle("title", fontName="Body-B", fontSize=15, leading=19.5, alignment=TA_CENTER, textColor=INK, spaceAfter=8),
    note=ParagraphStyle("note", fontName="Body", fontSize=8.3, leading=11.2, textColor=GREY, backColor=colors.HexColor("#f4f6f8"), borderPadding=(5, 6, 5, 6), spaceAfter=12, spaceBefore=2),
    h2=ParagraphStyle("h2", fontName="Body-B", fontSize=12.5, leading=16, textColor=ACC, spaceBefore=13, spaceAfter=5, keepWithNext=1),
    h3=ParagraphStyle("h3", fontName="Body-B", fontSize=10.3, leading=13.5, textColor=INK, spaceBefore=9, spaceAfter=3, keepWithNext=1),
    p=ParagraphStyle("p", fontName="Body", fontSize=9.2, leading=13, textColor=INK, spaceAfter=5, alignment=TA_LEFT),
    li=ParagraphStyle("li", fontName="Body", fontSize=9.2, leading=13, textColor=INK, spaceAfter=2),
    cell=ParagraphStyle("cell", fontName="Body", fontSize=7.4, leading=9.4, textColor=INK),
    cellh=ParagraphStyle("cellh", fontName="Body-B", fontSize=7.4, leading=9.4, textColor=INK),
    cap=ParagraphStyle("cap", fontName="Body-I", fontSize=8.0, leading=10.6, textColor=GREY, spaceBefore=2, spaceAfter=9),
)


def inline(t, code_size=8.3):
    toks = []

    def stash(m):
        toks.append(m.group(1)); return f"\x00{len(toks) - 1}\x00"
    t = re.sub(r"`([^`]+)`", stash, t)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*([^*\s][^*]*?)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: f'<font name="Code" size="{code_size}">' + toks[int(m.group(1))].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") + "</font>", t)
    return t


def build_table(rows, width):
    cells = [[Paragraph(inline(c, 7), S["cellh"] if i == 0 else S["cell"]) for c in r] for i, r in enumerate(rows)]
    n = max(len(r) for r in rows)
    cells = [r + [Paragraph("", S["cell"])] * (n - len(r)) for r in cells]
    w = []
    for j in range(n):
        m = max(len(re.sub(r"[*`]", "", r[j])) if j < len(r) else 0 for r in rows)
        w.append(min(max(m, 5), 46) + 3)
    tot = sum(w); cw = [width * x / tot for x in w]
    t = Table(cells, colWidths=cw, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), HEAD), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2)]))
    return t


class Doc(BaseDocTemplate):
    def __init__(self, fn, **kw):
        super().__init__(fn, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm, bottomMargin=1.9 * cm, **kw)
        fr = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([PageTemplate(id="p", frames=[fr], onPage=self.deco)]); self._k = 0

    def deco(self, c, d):
        c.saveState(); c.setFont("Body", 7.6); c.setFillColor(GREY)
        c.drawString(2 * cm, 1.1 * cm, "Ixchel — preprint v1.0 · CC BY 4.0 · laboratório de pesquisa e desenvolvimento"); c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{d.page}")
        c.setStrokeColor(LINE); c.setLineWidth(0.4); c.line(2 * cm, 1.5 * cm, A4[0] - 2 * cm, 1.5 * cm); c.restoreState()

    def afterFlowable(self, f):
        if isinstance(f, Paragraph) and f.style.name in ("h2", "h3"):
            lvl = 0 if f.style.name == "h2" else 1; txt = re.sub(r"<[^>]+>", "", f.text); key = f"h{self._k}"; self._k += 1
            self.canv.bookmarkPage(key); self.canv.addOutlineEntry(txt, key, level=lvl, closed=(lvl == 1))


def main():
    md = open(os.path.join(HERE, "PREPRINT.md"), encoding="utf-8").read().splitlines()
    out = os.path.join(HERE, "Ixchel_preprint.pdf"); doc = Doc(out, title="Ixchel: modelado de lenguaje por mezcla de expertos de conteo, sin gradientes ni hiperparámetros ajustados, en CPU pura", author="", subject="Borrador de preprint")
    W = doc.width; flow = []; i = 0; first_p = True
    while i < len(md):
        ln = md[i]
        if not ln.strip():
            i += 1; continue
        if ln.startswith("# "):
            flow.append(Paragraph(inline(ln[2:]), S["title"])); i += 1; continue
        if ln.startswith("### "):
            flow.append(Paragraph(inline(ln[4:]), S["h3"])); i += 1; continue
        if ln.startswith("## "):
            flow.append(CondPageBreak(3.2 * cm)); flow.append(Paragraph(inline(ln[3:]), S["h2"])); i += 1; continue
        m = re.match(r"!\[(.*)\]\((.*)\)\s*$", ln)
        if m:
            p = os.path.join(HERE, m.group(2)); from PIL import Image as PI
            iw, ih = PI.open(p).size; wd = min(W, 15.2 * cm); ht = wd * ih / iw
            flow.append(KeepTogether([Image(p, width=wd, height=ht), Paragraph(inline(m.group(1)), S["cap"])])); i += 1; continue
        if ln.startswith("|"):
            rows = []
            while i < len(md) and md[i].startswith("|"):
                cells = [c.strip() for c in md[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            flow.append(build_table(rows, W)); flow.append(Spacer(1, 7)); continue
        if re.match(r"^(- |\d+\. )", ln):
            items = []; ordered = bool(re.match(r"^\d+\. ", ln))
            while i < len(md) and re.match(r"^(- |\d+\. )", md[i]):
                items.append(ListItem(Paragraph(inline(re.sub(r"^(- |\d+\. )", "", md[i])), S["li"]), leftIndent=14)); i += 1
            flow.append(ListFlowable(items, bulletType="1" if ordered else "bullet", bulletFontName="Body", bulletFontSize=8.5, leftIndent=14, bulletOffsetY=0, start=1 if ordered else "•"))
            flow.append(Spacer(1, 4)); continue
        style = S["note"] if (first_p and ln.startswith("**Preprint v1.0")) else S["p"]; first_p = False if style is S["note"] else first_p
        flow.append(Paragraph(inline(ln), style)); i += 1
    doc.build(flow); print("PDF:", out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
