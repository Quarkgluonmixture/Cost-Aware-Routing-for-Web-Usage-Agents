#!/usr/bin/env bash
# convert.sh — Markdown section files -> ACL 2026 double-blind PDF.
#
# Derived from docs/checkpoints/paper_drafts/aaai27/latex/convert.sh (AAAI-27
# dry-run, kept intact because tests/test_dayaudit_rounda_20260714.py pins its
# behaviour with a fixture). Three differences:
#   1. venue template is the official ACL kit, not the AAAI author kit;
#   2. the source is a list of per-section Markdown files, not one main.md, so
#      the two REALM papers can share one converter;
#   3. build state is per paper (build/paperA, build/paperB), so converting one
#      paper cannot leave stale .aux/.bbl behind for the other.
#
# Usage: convert.sh <paperA|paperB> [--submission]
#   --submission  fail unless the generated TeX has zero visible TODO slots
#                 AND the references start no later than page 9 (= the 8-page
#                 REALM content limit is respected).

set -euo pipefail

PAPER=''
SUBMISSION=0
for argument in "$@"; do
  case "$argument" in
    paperA|paperB) PAPER="$argument" ;;
    --submission) SUBMISSION=1 ;;
    -h|--help)
      printf 'Usage: %s <paperA|paperB> [--submission]\n' "$0"
      exit 0
      ;;
    *)
      printf 'ERROR: unknown argument: %s\n' "$argument" >&2
      printf 'Usage: %s <paperA|paperB> [--submission]\n' "$0" >&2
      exit 2
      ;;
  esac
done
if [[ -z "$PAPER" ]]; then
  printf 'ERROR: paper id required (paperA or paperB)\n' >&2
  printf 'Usage: %s <paperA|paperB> [--submission]\n' "$0" >&2
  exit 2
fi

LATEX_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(git -C "$LATEX_DIR" rev-parse --show-toplevel)"
DRAFT_DIR="$LATEX_DIR/../$PAPER"
BIB_SOURCE="$LATEX_DIR/../paper.bib"
SKELETON="$LATEX_DIR/skeleton_acl.tex"
BUILD_DIR="$LATEX_DIR/build/$PAPER"
FIGURE_SOURCE="$REPO_ROOT/results/phantom_paper/figures"

# Per-paper facts. Section order is explicit rather than a glob: a new file
# dropped into the draft directory must be placed deliberately, not silently
# appended in lexical order.
#
# EXPECTED_TABLES is an assertion, not a setting. Pandoc emits longtable for
# every pipe table and longtable is illegal in two-column mode, so a table
# whose caption line is missing survives as longtable and is caught by the
# longtable guard; a table that is dropped entirely would not be, hence the
# count.
case "$PAPER" in
  paperA)
    SECTIONS=(
      section1_intro.md
      section2_setup.md
      section3_findings.md
      section4_behaviour.md
      section5_discussion.md
    )
    EXPECTED_TABLES=4
    FIGURES=(fig_f1_diamond_schematic.pdf fig_f2_h1_forest.pdf)
    ;;
  paperB)
    SECTIONS=(
      section1_intro.md
      section2_setup.md
      section3_ceiling.md
      section4_supply.md
      section5_triage.md
      section6_relabelling.md
      section7_discussion.md
    )
    EXPECTED_TABLES=12
    FIGURES=()
    ;;
esac

for command_name in pandoc latexmk pdflatex bibtex pdfinfo perl awk rg; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'ERROR: required command not found: %s\n' "$command_name" >&2
    exit 2
  fi
done

TITLE_FILE="$DRAFT_DIR/TITLE.md"
for input_file in "$TITLE_FILE" "$BIB_SOURCE" "$SKELETON"; do
  if [[ ! -f "$input_file" ]]; then
    printf 'ERROR: required input not found: %s\n' "$input_file" >&2
    exit 2
  fi
done
for section_file in "${SECTIONS[@]}"; do
  if [[ ! -f "$DRAFT_DIR/$section_file" ]]; then
    printf 'ERROR: section file not found: %s\n' "$DRAFT_DIR/$section_file" >&2
    exit 2
  fi
done
for figure_file in ${FIGURES[@]+"${FIGURES[@]}"}; do
  if [[ ! -f "$FIGURE_SOURCE/$figure_file" ]]; then
    printf 'ERROR: figure not found: %s\n' "$FIGURE_SOURCE/$figure_file" >&2
    exit 2
  fi
done

# build/ is generated state. Recreating it makes conversion idempotent and
# prevents stale .aux/.bbl files from hiding a broken citation or template.
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/figures"

# paper.bib doubles as a literature notebook: its note fields contain long
# internal triage prose that standard BibTeX styles print verbatim. Strip only
# the generated copy so the source remains untouched and references remain a
# realistic dry-run. All current note fields are deliberately one physical line.
perl -ne 'print unless /^\s*note\s*=/i' "$BIB_SOURCE" \
  > "$BUILD_DIR/paper.bib"
for figure_file in ${FIGURES[@]+"${FIGURES[@]}"}; do
  cp "$FIGURE_SOURCE/$figure_file" "$BUILD_DIR/figures/"
done

# Assemble the single-file master. The section files carry the abstract and
# body at heading levels 2 and 3 because they are read as standalone drafts;
# promoting one level here makes level 1 the section level pandoc maps to
# \section, and keeps the drafts free of build-only markup.
TITLE="$(awk 'NF { sub(/^[[:space:]]*#+[[:space:]]*/, ""); print; exit }' "$TITLE_FILE")"
if [[ -z "$TITLE" ]]; then
  printf 'ERROR: no title text found in %s\n' "$TITLE_FILE" >&2
  exit 2
fi
{
  printf -- '---\ntitle: "%s"\n---\n\n' "$TITLE"
  for section_file in "${SECTIONS[@]}"; do
    cat "$DRAFT_DIR/$section_file"
    printf '\n'
  done
} | perl -Mutf8 -CSDA -pe 's{^\#(\#+[ \t])}{$1}' > "$BUILD_DIR/assembled.md"

if ! rg -q '^# Abstract' "$BUILD_DIR/assembled.md"; then
  printf 'ERROR: assembled master has no level-1 "# Abstract" heading\n' >&2
  exit 2
fi

# Work only on a generated copy: remove status/checklist HTML comments,
# normalize numbered Markdown headings, expose data slots as raw LaTeX TODOs,
# and disambiguate currency dollars from inline math. The heading-number
# pattern accepts a trailing period ("## 1. Introduction"), which the AAAI
# converter did not need because its master used "# 1 Introduction".
perl -Mutf8 -CSDA -0777 -pe '
  s{<!--.*?-->}{}gs;
  s{^(\#{1,6})[ \t]+(?:[0-9]+(?:\.[0-9]+)*\.?)[ \t]+}{$1 }mg;
  s{(?:\\)?<((?:H[0-9]+-VERDICT|TBD|TODO))(?:\s*:[^>\n]*)?(?:\\)?>}{\\todo{$1}}g;
  s{«[^»\n]+»}{\\todo{SLOT}}g;
  s{⟨[^⟩\n]+⟩}{\\todo{SLOT}}g;
  s{\$(?=\d)}{\\\$}g;
' "$BUILD_DIR/assembled.md" > "$BUILD_DIR/source_sanitized.md"

awk -v abstract_out="$BUILD_DIR/abstract.md" -v body_out="$BUILD_DIR/body.md" '
  $0 == "# Abstract" { mode = "abstract"; next }
  mode == "abstract" && /^# / { mode = "body" }
  mode == "abstract" { print > abstract_out; next }
  mode == "body" { print > body_out }
' "$BUILD_DIR/source_sanitized.md"

if [[ ! -s "$BUILD_DIR/abstract.md" || ! -s "$BUILD_DIR/body.md" ]]; then
  printf 'ERROR: failed to split Abstract/body at Markdown level-1 headings\n' >&2
  exit 2
fi

PANDOC_ARGS=(
  -f markdown+pipe_tables+raw_tex+tex_math_dollars
  -t latex
  --natbib
  --wrap=none
)

printf '%s\n' "$TITLE" | pandoc "${PANDOC_ARGS[@]}" \
  > "$BUILD_DIR/title_generated.tex"
pandoc "${PANDOC_ARGS[@]}" "$BUILD_DIR/abstract.md" \
  > "$BUILD_DIR/abstract_generated.tex"
pandoc "${PANDOC_ARGS[@]}" "$BUILD_DIR/body.md" \
  > "$BUILD_DIR/body_pandoc.tex"

# Pandoc intentionally emits longtable. longtable is illegal in two-column
# mode, so turn each pipe table plus its adjacent italic "Table N" caption line
# into a booktabs table* float. The Markdown captions remain the single source
# of caption prose; no LaTeX edit is required.
#
# table* (two-column-spanning) rather than table is deliberate. A single-column
# variant was measured on 2026-07-27: it saves one page but the four-column
# tables whose headers are plain `l` columns then overrun the column and print
# on top of the neighbouring text, with no Overfull warning to catch it. Width
# safety beats one page; the page budget is managed by the appendix instead.
cat > "$BUILD_DIR/floatify.pl" <<'FLOATIFY'
local $/;
my $t = <STDIN>;
$t =~ s{\{\\def\\LTcaptype\{none\}\s*%[^\n]*\n}{}g;
$t =~ s{(\\end\{longtable\})\s*\n\}}{$1}g;
$t =~ s{\\begin\{longtable\}\[\]}{\\begin{tabular}}g;
$t =~ s{\\end\{longtable\}}{\\end{tabular}}g;
$t =~ s{\\endhead\s*}{}g;
$t =~ s{\\endlastfoot\s*}{}g;

sub floatify {
  my ($tabular, $caption) = @_;
  return "\\begin{table*}[t]\n\\centering\n\\caption{$caption}\n\\small\n"
       . "\\setlength{\\tabcolsep}{4pt}\n$tabular\n\\end{table*}\n\n";
}

$t =~ s{(\\begin\{tabular\}.*?\\end\{tabular\})\s*\\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*}
       {floatify($1, $2)}gsex;
$t =~ s{\\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*(\\begin\{tabular\}.*?\\end\{tabular\})\s*}
       {floatify($2, $1)}gsex;
print $t;
FLOATIFY

perl "$BUILD_DIR/floatify.pl" \
  < "$BUILD_DIR/body_pandoc.tex" > "$BUILD_DIR/body_generated.tex"

# Optional appendix. REALM does not count references or appendices against the
# 8-page content limit, so supporting tables live here rather than being cut.
# It goes through the identical pandoc + float pipeline as the body, so a table
# does not change shape by being moved across the boundary.
APPENDIX_SOURCE="$DRAFT_DIR/appendix.md"
APPENDIX_INPUT=''
if [[ -f "$APPENDIX_SOURCE" ]]; then
  perl -Mutf8 -CSDA -0777 -pe '
    s{<!--.*?-->}{}gs;
    s{^\#(\#+[ \t])}{$1}mg;
    s{^(\#{1,6})[ \t]+(?:[A-Z]?[0-9]*(?:\.[0-9]+)*\.?)[ \t]+}{$1 }mg;
    s{\$(?=\d)}{\\\$}g;
  ' "$APPENDIX_SOURCE" > "$BUILD_DIR/appendix_sanitized.md"
  pandoc "${PANDOC_ARGS[@]}" "$BUILD_DIR/appendix_sanitized.md" \
    > "$BUILD_DIR/appendix_pandoc.tex"
  perl "$BUILD_DIR/floatify.pl" \
    < "$BUILD_DIR/appendix_pandoc.tex" > "$BUILD_DIR/appendix_generated.tex"
  APPENDIX_INPUT='\appendix\input{appendix_generated.tex}'
  if rg -q '\\begin\{longtable\}' "$BUILD_DIR/appendix_generated.tex"; then
    printf 'ERROR: longtable survived in the appendix (a table is missing its caption line)\n' >&2
    exit 2
  fi
fi

# Figures are inserted by section anchor rather than by Markdown image syntax:
# the drafts are read as prose by three reviewers and by Vale, and a float
# placement directive is build state, not prose.
if [[ "$PAPER" == 'paperA' ]]; then
  perl -0777 -pe '
    s{(\\section\{The phantom routing space\}[^\n]*\n)}{$1
\\begin{figure}[t]
\\centering
\\includegraphics[width=\\columnwidth]{fig_f1_diamond_schematic.pdf}
\\caption{The \$2\\times2\$ ablation diamond over text payload and prompt family. The screenshot is off in all four cells; the two image-bearing modes bound the space from outside.}
\\label{fig:diamond}
\\end{figure}
}s;
    s{(\\section\{Results\}[^\n]*\n)}{$1
\\begin{figure}[t]
\\centering
\\includegraphics[width=\\columnwidth]{fig_f2_h1_forest.pdf}
\\caption{Per-cell P-SoM strict six-mode drop-one contributions with the fixed-effects pooled estimate against the preregistered \$+1.0\$\\,pp threshold.}
\\label{fig:h1forest}
\\end{figure}
}s;
  ' "$BUILD_DIR/body_generated.tex" > "$BUILD_DIR/body_with_figures.tex"
  mv "$BUILD_DIR/body_with_figures.tex" "$BUILD_DIR/body_generated.tex"
  for label in fig:diamond fig:h1forest; do
    if ! rg -q "\\\\label\\{$label\\}" "$BUILD_DIR/body_generated.tex"; then
      printf 'ERROR: figure anchor missing, %s was not inserted\n' "$label" >&2
      exit 2
    fi
  done
fi

# A manuscript with no \cite makes bibtex emit an empty thebibliography, and
# LaTeX then dies on "Something's wrong--perhaps a missing \item" from main.bbl,
# several hundred log lines away from the actual cause. Say it here instead.
TEX_PARTS=("$BUILD_DIR/body_generated.tex" "$BUILD_DIR/abstract_generated.tex")
[[ -f "$BUILD_DIR/appendix_generated.tex" ]] && TEX_PARTS+=("$BUILD_DIR/appendix_generated.tex")
CITE_COUNT="$({ rg -o '\\cite[a-zA-Z]*\{' "${TEX_PARTS[@]}" || true; } | wc -l)"
if [[ "$CITE_COUNT" -eq 0 ]]; then
  printf 'ERROR: no \\cite commands in the generated TeX. bibtex would emit an empty\n' >&2
  printf '       bibliography and pdflatex would fail inside main.bbl. Add citations to\n' >&2
  printf '       the Markdown sections (pandoc --natbib turns [@key] into \\citep{key}).\n' >&2
  exit 2
fi

if rg -q '\\begin\{longtable\}|\\end\{longtable\}' "$BUILD_DIR/body_generated.tex"; then
  printf 'ERROR: longtable survived two-column post-processing (a table is missing its "*Table N: ...*" caption line)\n' >&2
  rg -n '\\begin\{longtable\}' "$BUILD_DIR/body_generated.tex" >&2 || true
  exit 2
fi
# Counted over body plus appendix, so moving a table across that boundary to
# manage the page budget does not silently change what the assertion covers.
ACTUAL_TABLES="$({ rg -o '\\begin\{table\*\}' "${TEX_PARTS[@]}" || true; } | wc -l)"
if [[ "${ACTUAL_TABLES:-0}" -ne "$EXPECTED_TABLES" ]]; then
  printf 'ERROR: expected %s converted table* environments across body+appendix, found %s\n' \
    "$EXPECTED_TABLES" "${ACTUAL_TABLES:-0}" >&2
  exit 2
fi

# Official kit detection. The proxy fallback exists so a machine without the
# kit still produces a readable PDF; its page count is explicitly not
# authoritative, which the template version string records inside the PDF.
if [[ -f "$LATEX_DIR/acl.sty" && -f "$LATEX_DIR/acl_natbib.bst" ]]; then
  VENUE_SETUP='\usepackage[review]{acl}'
  BIB_STYLE='acl_natbib'
  TEMPLATE_VERSION='ACL 2026 official style files (review/double-blind)'
  BUILD_MODE='official-acl2026'
  cp "$LATEX_DIR/acl.sty" "$LATEX_DIR/acl_natbib.bst" "$BUILD_DIR/"
else
  VENUE_SETUP='\usepackage[margin=0.75in,columnsep=0.25in]{geometry}\twocolumn'
  BIB_STYLE='plainnat'
  TEMPLATE_VERSION='PROXY article two-column; page count is non-authoritative'
  BUILD_MODE='proxy-two-column'
fi

VENUE_SETUP_SED="${VENUE_SETUP//\\/\\\\}"
APPENDIX_INPUT_SED="${APPENDIX_INPUT//\\/\\\\}"
sed \
  -e "s|@@VENUE_SETUP@@|$VENUE_SETUP_SED|g" \
  -e "s|@@BIB_STYLE@@|$BIB_STYLE|g" \
  -e "s|@@TEMPLATE_VERSION@@|$TEMPLATE_VERSION|g" \
  -e "s|@@APPENDIX_INPUT@@|$APPENDIX_INPUT_SED|g" \
  "$SKELETON" > "$BUILD_DIR/main.tex"
printf '%s\n' "$BUILD_MODE" > "$BUILD_DIR/build_mode.txt"

(
  cd "$BUILD_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
) 2>&1 | tee "$BUILD_DIR/latexmk.stdout.log"

PAGES="$(pdfinfo "$BUILD_DIR/main.pdf" | awk '/^Pages:/ { print $2 }')"
TODO_COUNT="$({ rg -o '\\todo\{' "$BUILD_DIR/abstract_generated.tex" "$BUILD_DIR/body_generated.tex" || true; } | wc -l)"
REF_PAGE="$(sed -n 's/.*\\newlabel{refs-start}{{[^}]*}{\([^}]*\)}.*/\1/p' "$BUILD_DIR/main.aux" | tail -n 1)"
UNDEF_CITES="$({ rg -o 'Citation .* undefined' "$BUILD_DIR/main.log" || true; } | wc -l)"

printf '\nPaper: %s\n' "$PAPER"
printf 'Build mode: %s\n' "$BUILD_MODE"
printf 'PDF: %s (%s pages)\n' "$BUILD_DIR/main.pdf" "$PAGES"
printf 'Reference start page: %s (REALM content limit is 8)\n' "${REF_PAGE:-unknown}"
printf 'Visible TODO slots: %s\n' "$TODO_COUNT"
printf 'Undefined citations: %s\n' "$UNDEF_CITES"

if [[ "$SUBMISSION" -eq 1 ]]; then
  SUBMISSION_FAILED=0
  if [[ "$TODO_COUNT" -ne 0 ]]; then
    printf 'ERROR: --submission requires zero visible TODO slots; residuals:\n' >&2
    rg -n '\\todo\{' "$BUILD_DIR/abstract_generated.tex" "$BUILD_DIR/body_generated.tex" >&2 || true
    SUBMISSION_FAILED=1
  fi
  if [[ "$UNDEF_CITES" -ne 0 ]]; then
    printf 'ERROR: --submission requires zero undefined citations\n' >&2
    rg -n 'Citation .* undefined' "$BUILD_DIR/main.log" >&2 || true
    SUBMISSION_FAILED=1
  fi
  if [[ -z "${REF_PAGE:-}" ]]; then
    printf 'ERROR: --submission could not locate the reference start page\n' >&2
    SUBMISSION_FAILED=1
  elif [[ "$REF_PAGE" -gt 9 ]]; then
    printf 'ERROR: --submission requires content within 8 pages; references start on page %s\n' \
      "$REF_PAGE" >&2
    SUBMISSION_FAILED=1
  fi
  [[ "$SUBMISSION_FAILED" -eq 0 ]] || exit 2
fi
