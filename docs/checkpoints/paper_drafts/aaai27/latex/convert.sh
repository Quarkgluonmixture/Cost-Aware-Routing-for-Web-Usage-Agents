#!/usr/bin/env bash
set -euo pipefail

SUBMISSION=0
if [[ $# -gt 1 ]]; then
  printf 'Usage: %s [--submission]\n' "$0" >&2
  exit 2
fi
case "${1:-}" in
  "") ;;
  --submission) SUBMISSION=1 ;;
  -h|--help)
    printf 'Usage: %s [--submission]\n' "$0"
    printf '  --submission  fail unless the generated TeX contains zero visible TODO slots\n'
    exit 0
    ;;
  *)
    printf 'ERROR: unknown argument: %s\n' "$1" >&2
    printf 'Usage: %s [--submission]\n' "$0" >&2
    exit 2
    ;;
esac

LATEX_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(git -C "$LATEX_DIR" rev-parse --show-toplevel)"
SOURCE_MD="$LATEX_DIR/../aaai27_main.md"
BIB_SOURCE="$LATEX_DIR/../../paper.bib"
SKELETON="$LATEX_DIR/skeleton.tex"
BUILD_DIR="$LATEX_DIR/build"
FIGURE_SOURCE="$REPO_ROOT/results/phantom_paper/figures"

for command_name in pandoc latexmk pdflatex bibtex pdfinfo perl awk; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'ERROR: required command not found: %s\n' "$command_name" >&2
    exit 2
  fi
done

for input_file in "$SOURCE_MD" "$BIB_SOURCE" "$SKELETON" \
  "$FIGURE_SOURCE/fig_f1_diamond_schematic.pdf" \
  "$FIGURE_SOURCE/fig_f2_h1_forest.pdf"; do
  if [[ ! -f "$input_file" ]]; then
    printf 'ERROR: required input not found: %s\n' "$input_file" >&2
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
cp "$FIGURE_SOURCE/fig_f1_diamond_schematic.pdf" "$BUILD_DIR/figures/"
cp "$FIGURE_SOURCE/fig_f2_h1_forest.pdf" "$BUILD_DIR/figures/"

# Work only on a generated copy: remove status/checklist HTML comments,
# normalize numbered Markdown headings, expose data slots as raw LaTeX TODOs,
# and disambiguate currency dollars from inline math.
perl -Mutf8 -CSDA -0777 -pe '
  s{<!--.*?-->}{}gs;
  s{^(#{1,6})[ \t]+(?:[0-9]+(?:\.[0-9]+)*)[ \t]+}{$1 }mg;
  s{(?:\\)?<((?:H[0-9]+-VERDICT|TBD|TODO))(?:\s*:[^>\n]*)?(?:\\)?>}{\\todo{$1}}g;
  s{«[^»\n]+»}{\\todo{SLOT}}g;
  s{⟨[^⟩\n]+⟩}{\\todo{SLOT}}g;
  s{\$(?=\d)}{\\\$}g;
' "$SOURCE_MD" > "$BUILD_DIR/source_sanitized.md"

TITLE="$({
  awk '
    /^---[[:space:]]*$/ { in_yaml = !in_yaml; next }
    in_yaml && /^title:[[:space:]]*/ {
      sub(/^title:[[:space:]]*/, "");
      sub(/^"/, ""); sub(/"[[:space:]]*$/, "");
      print; exit
    }
  ' "$BUILD_DIR/source_sanitized.md"
} || true)"
if [[ -z "$TITLE" ]]; then
  printf 'ERROR: YAML title not found in %s\n' "$SOURCE_MD" >&2
  exit 2
fi

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
# mode, so turn each pipe table plus its following italic "Table N" line into
# a booktabs table* float. The source captions remain the single source of
# caption prose; no Markdown edit is required.
perl -0777 -pe '
  s{\{\\def\\LTcaptype\{none\}\s*%[^\n]*\n}{}g;
  s{(\\end\{longtable\})\s*\n\}}{$1}g;
  s{\\begin\{longtable\}\[\]}{\\begin{tabular}}g;
  s{\\end\{longtable\}}{\\end{tabular}}g;
  s{\\endhead\s*}{}g;
  s{\\endlastfoot\s*}{}g;
  s{
    (\\begin\{tabular\}.*?\\end\{tabular\})\s*
    \\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*
  }{
\\begin{table*}[t]
\\centering
\\caption{$2}
\\small
\\setlength{\\tabcolsep}{4pt}
$1
\\end{table*}

  }gsx;
  s{
    \\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*
    (\\begin\{tabular\}.*?\\end\{tabular\})\s*
  }{
\\begin{table*}[t]
\\centering
\\caption{$1}
\\small
\\setlength{\\tabcolsep}{4pt}
$2
\\end{table*}

  }gsx;
  s{(\\section\{Introduction\}[^\n]*\n)}{$1
\\begin{figure}[t]
\\centering
\\includegraphics[width=\\columnwidth]{fig_f1_diamond_schematic.pdf}
\\caption{The \$2\\times2\$ ablation diamond over text-payload format and prompt family.}
\\label{fig:diamond}
\\end{figure}
}s;
  s{(\\section\{Results I: The Phenomenon\}[^\n]*\n)}{$1
\\begin{figure}[t]
\\centering
\\includegraphics[width=\\columnwidth]{fig_f2_h1_forest.pdf}
\\caption{Per-cell P-SoM strict six-mode drop-one results with the fixed-effects pooled estimate against the pre-registered \$+1.0\$pp threshold.}
\\label{fig:h1forest}
\\end{figure}
}s;
' "$BUILD_DIR/body_pandoc.tex" > "$BUILD_DIR/body_generated.tex"

if rg -q '\\begin\{longtable\}|\\end\{longtable\}' "$BUILD_DIR/body_generated.tex"; then
  printf 'ERROR: longtable survived two-column post-processing\n' >&2
  exit 2
fi
if [[ "$(rg -c '\\begin\{table\*\}' "$BUILD_DIR/body_generated.tex")" -ne 4 ]]; then
  printf 'ERROR: expected four converted table* environments\n' >&2
  exit 2
fi

if [[ -f "$LATEX_DIR/aaai27.sty" && -f "$LATEX_DIR/aaai27.bst" ]]; then
  CLASS_OPTIONS='letterpaper'
  VENUE_SETUP='\usepackage[submission]{aaai27}'
  BIB_STYLE='aaai27'
  TEMPLATE_VERSION='AAAI-27 official author kit'
  BUILD_MODE='official-aaai27'
  cp "$LATEX_DIR/aaai27.sty" "$LATEX_DIR/aaai27.bst" "$BUILD_DIR/"
else
  CLASS_OPTIONS='10pt,letterpaper,twocolumn'
  VENUE_SETUP='\usepackage[margin=0.75in,columnsep=0.25in]{geometry}'
  BIB_STYLE='plainnat'
  TEMPLATE_VERSION='PROXY article two-column; page count is non-authoritative'
  BUILD_MODE='proxy-two-column'
fi

VENUE_SETUP_SED="${VENUE_SETUP//\\/\\\\}"
sed \
  -e "s|@@CLASS_OPTIONS@@|$CLASS_OPTIONS|g" \
  -e "s|@@VENUE_SETUP@@|$VENUE_SETUP_SED|g" \
  -e "s|@@BIB_STYLE@@|$BIB_STYLE|g" \
  -e "s|@@TEMPLATE_VERSION@@|$TEMPLATE_VERSION|g" \
  "$SKELETON" > "$BUILD_DIR/main.tex"
printf '%s\n' "$BUILD_MODE" > "$BUILD_DIR/build_mode.txt"

(
  cd "$BUILD_DIR"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
) 2>&1 | tee "$BUILD_DIR/latexmk.stdout.log"

PAGES="$(pdfinfo "$BUILD_DIR/main.pdf" | awk '/^Pages:/ { print $2 }')"
TODO_COUNT="$({ rg -o '\\todo\{' "$BUILD_DIR/abstract_generated.tex" "$BUILD_DIR/body_generated.tex" || true; } | wc -l)"
REF_PAGE="$(sed -n 's/.*\\newlabel{refs-start}{{[^}]*}{\([^}]*\)}.*/\1/p' "$BUILD_DIR/main.aux" | tail -n 1)"

printf '\nBuild mode: %s\n' "$BUILD_MODE"
printf 'PDF: %s (%s pages)\n' "$BUILD_DIR/main.pdf" "$PAGES"
printf 'Reference start page: %s\n' "${REF_PAGE:-unknown}"
printf 'Visible TODO slots: %s\n' "$TODO_COUNT"

if [[ "$SUBMISSION" -eq 1 && "$TODO_COUNT" -ne 0 ]]; then
  printf 'ERROR: --submission requires zero visible TODO slots; residuals:\n' >&2
  rg -n '\\todo\{' "$BUILD_DIR/abstract_generated.tex" "$BUILD_DIR/body_generated.tex" >&2 || true
  exit 2
fi
