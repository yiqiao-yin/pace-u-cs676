"""Render the CS676 slide PDF into one standalone HTML presentation app."""
import base64, io, json, re, sys, time
from pathlib import Path
import pypdfium2 as pdfium
from pypdf import PdfReader

PDF = Path("docs/slide_doc/CS 676 Algorithms of Data Science.pdf")
OUT = Path(sys.argv[1])
SCALE, QUALITY = 1.8, 72

BOILER = ("https://www.youtube.com/YiqiaoYin", "https://www.y-yin.io/", "Back to top.",
          "Data Science, Machine Learning, Artificial Intelligence, GenAI",
          "Data Science, Machine Learning, Artiﬁcial Intelligence, GenAI",
          "Go back to main content, click here")

LIG = {"\ufb00":"ff","\ufb01":"fi","\ufb02":"fl","\ufb03":"ffi","\ufb04":"ffl",
       "\u201c":'"',"\u201d":'"',"\u2018":"'","\u2019":"'"}

def delig(s):
    for k, v in LIG.items():
        s = s.replace(k, v)
    return s

def clean(raw):
    raw = delig(raw or "")
    out = []
    for l in (raw or "").splitlines():
        l = l.strip()
        if l and not any(b in l for b in BOILER):
            out.append(l)
    return out

# ---- text layer: titles + searchable body -----------------------------------
reader = PdfReader(str(PDF))
texts = [clean(p.extract_text()) for p in reader.pages]

CHAPTERS = [(2,"Introduction"),(30,"Basics in Statistical Learning"),(78,"Linear Regression"),
            (99,"Classification"),(121,"Sampling and Bootstrap"),(130,"Model Selection & Regularization"),
            (153,"Going Beyond Linearity"),(159,"Tree-based Methods"),(191,"Support Vector Machine"),
            (207,"Deep Learning"),(275,"Unsupervised Learning"),(308,"Classification Metrics")]

def chapter_of(page):                      # 1-indexed page -> chapter index
    idx = 0
    for i,(start,_) in enumerate(CHAPTERS):
        if page >= start: idx = i
    return idx if page >= CHAPTERS[0][0] else 0

def title_of(page, lines):
    """Best available label for the TOC — chapter name on dividers, else the
    opening line trimmed to something that fits a sidebar row."""
    if page == 1:
        return "Title slide"
    chap = next((l for l in lines if re.match(r'^Chapter \d+', l)), None)
    body = [l for l in lines if not re.match(r'^Chapter \d+', l)]
    if chap and not body:                       # a pure chapter divider
        return chap.replace(" - ", " — ")
    if body:                                    # the opening line represents the slide
        first = body[0]
        return (first[:57].rstrip() + "…") if len(first) > 60 else first
    if chap:
        return chap.replace(" - ", " — ")
    return f"Figure · {CHAPTERS[chapter_of(page)][1]}"   # image-only page

# ---- render -----------------------------------------------------------------
doc = pdfium.PdfDocument(str(PDF))
slides, total, t0 = [], 0, time.time()
for i in range(len(doc)):
    img = doc[i].render(scale=SCALE).to_pil().convert("RGB")
    buf = io.BytesIO(); img.save(buf, "WEBP", quality=QUALITY, method=6)
    b = buf.getvalue(); total += len(b)
    page = i + 1
    lines = texts[i]
    slides.append({
        "n": page,
        "c": chapter_of(page),
        "t": title_of(page, lines),
        "x": " ".join(lines)[:1200],
        "img": base64.b64encode(b).decode("ascii"),
    })
    if page % 60 == 0:
        print(f"  {page}/{len(doc)}  {total/1e6:.1f} MB  {time.time()-t0:.0f}s", flush=True)

payload = {"chapters": [{"start": s, "name": n} for s, n in CHAPTERS], "slides": slides}
tpl = Path(sys.argv[2]).read_text(encoding="utf-8")
OUT.write_text(tpl.replace("/*__PAYLOAD__*/", json.dumps(payload, separators=(",", ":"))), encoding="utf-8")
print(f"images {total/1e6:.1f} MB | html {OUT.stat().st_size/1e6:.1f} MB | {time.time()-t0:.0f}s")
