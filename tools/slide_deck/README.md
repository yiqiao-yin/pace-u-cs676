# Slide deck generator

Turns `docs/slide_doc/CS 676 Algorithms of Data Science.pdf` into a single
self-contained HTML presentation app, deployed at
<https://main.d3j8dqgo1nf8ma.amplifyapp.com>.

## Why page images rather than extracted text

The equations in the PDF are **images, not text**. Page 84's text layer reads
"a linear model can be build as the following" and then stops — the formula
simply is not in the text. Rebuilding slides from extracted text would silently
drop every formula in the deck, so each page is rendered instead. Extracted text
is still used, for slide titles, the table of contents, and search.

## Rebuild

```bash
pip install pypdfium2 pypdf pillow
python tools/slide_deck/build_deck.py out.html tools/slide_deck/deck_template.html
```

Takes about 30 seconds and produces a ~19 MB file (330 WebP page renders, base64
inlined). **The output is intentionally not committed** — it is a build artifact
and would bloat every clone.

## Redeploy to Amplify

```bash
APP=d3j8dqgo1nf8ma
python -c "import zipfile;zipfile.ZipFile('site.zip','w',zipfile.ZIP_DEFLATED).write('out.html','index.html')"
R=$(aws amplify create-deployment --region us-east-1 --app-id $APP --branch-name main)
JOB=$(echo "$R" | python -c 'import sys,json;print(json.load(sys.stdin)["jobId"])')
URL=$(echo "$R" | python -c 'import sys,json;print(json.load(sys.stdin)["zipUploadUrl"])')
curl -X PUT -H "Content-Type: application/zip" --upload-file site.zip "$URL"
aws amplify start-deployment --region us-east-1 --app-id $APP --branch-name main --job-id "$JOB"
```
