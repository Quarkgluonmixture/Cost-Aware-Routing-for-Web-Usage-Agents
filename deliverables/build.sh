#!/usr/bin/env bash
# Build the restructured REALM submission from `main_restructured.tex`.
#
# Differs from `paper_drafts/latex/convert.sh` in one structural way: that script
# concatenates every section into ONE body and injects it into a skeleton, whereas
# `main_restructured.tex` owns its own `\input{sections/...}` list. So this script
# produces one .tex PER section and never rewrites the master file — the storyline
# lives in the master, and editing it must not require re-deriving anything.
#
# Tables are split, not duplicated. Seven go in the body, one per storyline beat;
# every other table lands in the appendix. The split is declared once, in
# BODY_TABLES below, and the appendix is its complement computed at build time —
# so a table can never be silently in both or in neither, which is the failure a
# hand-maintained pair of lists produces.
#
# Regenerate the table source first if products changed:
#   .venv/bin/python3 scripts/analysis/export_ablation_tables.py \
#     --evidence docs/checkpoints/paper_drafts/realm/section_evidence.md
#
# Usage:  bash deliverables/build.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DRAFT="$REPO/docs/checkpoints/paper_drafts/realm"
EVIDENCE="$DRAFT/section_evidence.md"
SECTIONS="$HERE/sections"
BUILD="$HERE/build"

# One table per storyline beat. Order is document order, not importance.
#   sr          the six modes and eight cells exist and differ            -> 2_setup
#   nonsep      WHICH cut produces the difference (the negative control)  -> 3_complementarity
#   failmode    the two channels fail differently                         -> 3_complementarity
#   floor       what a rerun of one arm buys, i.e. the correction         -> 3_noise
#   ceiling     what a perfect per-task choice could buy                  -> 4_upperbound
#   routing     five routing policies against the fixed ones              -> 5_lowerbound
#   instability why the gap is not closable with available supervision    -> 6_gap
declare -A BODY_TABLE_SECTION=(
  [sr]=2_setup
  [nonsep]=3_complementarity
  [failmode]=3_complementarity
  [floor]=3_noise
  [ceiling]=4_upperbound
  [routing]=5_lowerbound
  [instability]=6_gap
)

for c in pandoc latexmk pdflatex bibtex perl awk; do
  command -v "$c" >/dev/null || { echo "ERROR: $c not found" >&2; exit 2; }
done
[[ -f "$EVIDENCE" ]] || { echo "ERROR: $EVIDENCE missing — run export_ablation_tables.py --evidence" >&2; exit 2; }

mkdir -p "$SECTIONS" "$BUILD"
rm -f "$BUILD"/*.md "$BUILD"/table_*.md

PANDOC_ARGS=(-f markdown+pipe_tables+raw_tex+tex_math_dollars -t latex --natbib --wrap=none)

# --- floatify: pandoc emits longtable for every pipe table, and longtable is illegal
# in two-column mode. `table*` is the default because it is always width-safe; a
# single-column `table` is cheaper in pages but a long header can run past the column
# edge WITHOUT an Overfull warning, printing over the neighbouring text. Copied from
# convert.sh rather than imported so the two builds cannot drift apart silently.
cat > "$BUILD/floatify.pl" <<'FLOATIFY'
local $/;
my $t = <STDIN>;
my %single = map { $_ => 1 } split /\s+/, ($ENV{SINGLE_COL} // '');
my $seen = 0;
$t =~ s{\{\\def\\LTcaptype\{none\}\s*%[^\n]*\n}{}g;
$t =~ s{(\\end\{longtable\})\s*\n\}}{$1}g;
$t =~ s{\\begin\{longtable\}\[\]}{\\begin{tabular}}g;
$t =~ s{\\end\{longtable\}}{\\end{tabular}}g;
$t =~ s{\\endhead\s*}{}g;
$t =~ s{\\endlastfoot\s*}{}g;
sub floatify {
  my ($tabular, $caption) = @_;
  $seen++;
  my ($env, $place, $size, $sep) = $single{$seen}
    ? ("table",  "[tbp]", "\\footnotesize", "3pt")
    : ("table*", "[t]",   "\\small",        "4pt");
  return "\\begin{$env}$place\n\\centering\n\\caption{$caption}\n$size\n"
       . "\\setlength{\\tabcolsep}{$sep}\n$tabular\n\\end{$env}\n\n";
}
$t =~ s{(\\begin\{tabular\}.*?\\end\{tabular\})\s*\\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*}
       {floatify($1, $2)}gsex;
$t =~ s{\\emph\{Table\s+[0-9]+:\s*([^\n]*)\}\s*(\\begin\{tabular\}.*?\\end\{tabular\})\s*}
       {floatify($2, $1)}gsex;
print $t;
FLOATIFY

# --- extract one table block (markers are written by export_ablation_tables.py)
extract_table() {  # $1 = slug
  awk -v s="$1" '
    $0 == "<!-- BEGIN table:" s " -->" { on = 1; next }
    $0 == "<!-- END table:" s " -->"   { on = 0 }
    on { print }
  ' "$EVIDENCE"
}

ALL_SLUGS=$(grep -oE '<!-- BEGIN table:[a-z0-9-]+ -->' "$EVIDENCE" \
            | sed 's/<!-- BEGIN table://;s/ -->//')
N_ALL=$(printf '%s\n' "$ALL_SLUGS" | grep -c .)

# Every body slug must exist in the source. A typo would otherwise produce an empty
# table and a silently shorter paper.
for slug in "${!BODY_TABLE_SECTION[@]}"; do
  printf '%s\n' "$ALL_SLUGS" | grep -qx "$slug" \
    || { echo "ERROR: body table '$slug' not in $EVIDENCE" >&2; exit 2; }
  [[ -n "$(extract_table "$slug")" ]] \
    || { echo "ERROR: body table '$slug' extracted empty" >&2; exit 2; }
done

md_to_tex() {  # stdin -> stdout, one section's worth
  pandoc "${PANDOC_ARGS[@]}" | SINGLE_COL="" perl "$BUILD/floatify.pl"
}

# --- prose sections: markdown skeletons live beside the .tex they generate
for src in "$SECTIONS"/*.md; do
  [[ -e "$src" ]] || continue
  name="$(basename "$src" .md)"
  out="$SECTIONS/$name.tex"
  {
    cat "$src"
    # append this section's tables, in the order declared above
    for slug in sr nonsep failmode floor ceiling routing instability; do
      [[ "${BODY_TABLE_SECTION[$slug]:-}" == "$name" ]] || continue
      echo; echo; extract_table "$slug"
    done
  } | md_to_tex > "$out"
done

# --- appendix: the complement, computed rather than listed
{
  echo "## Appendix: full evidence tables"
  echo
  echo "The seven tables in the body are one per claim. Every other table this study"
  echo "produced is here, so a reader can check a number without taking a pointer on"
  echo "trust. Tables are grouped by what they measure."
  n_app=0
  for slug in $ALL_SLUGS; do
    [[ -n "${BODY_TABLE_SECTION[$slug]:-}" ]] && continue
    echo; echo; extract_table "$slug"
    n_app=$((n_app + 1))
  done
  echo "$n_app" > "$BUILD/.n_appendix"
} | md_to_tex > "$SECTIONS/appendix.tex"

N_BODY=${#BODY_TABLE_SECTION[@]}
N_APP=$(cat "$BUILD/.n_appendix")
if (( N_BODY + N_APP != N_ALL )); then
  echo "ERROR: $N_BODY body + $N_APP appendix != $N_ALL source tables" >&2
  exit 2
fi

# --- compile
cd "$HERE"
latexmk -pdf -quiet -interaction=nonstopmode \
  -outdir=build main_restructured.tex >/dev/null 2>&1 || true
latexmk -pdf -quiet -interaction=nonstopmode -outdir=build main_restructured.tex \
  > "$BUILD/latexmk.log" 2>&1 || {
    echo "ERROR: latexmk failed — see $BUILD/latexmk.log" >&2
    grep -m5 -E '^!' "$BUILD/main.log" 2>/dev/null >&2 || true
    exit 1; }

PAGES=$(pdfinfo build/main_restructured.pdf 2>/dev/null | awk '/^Pages:/{print $2}')
# grep -c prints 0 AND exits 1 when there are no matches. `|| echo 0` would append a
# SECOND line ("0\n0"); piping to head -1 fixes the text but trips pipefail, which
# under set -e killed the script before it printed anything. `|| true` keeps grep's
# own "0" and swallows only the exit status.
UNDEF=$(grep -c "Citation.*undefined" build/main_restructured.log 2>/dev/null || true)
CONTENT_END=$(perl -ne 'print "$2\n" if /\{content-end\}\{\{([^}]*)\}\{(\d+)\}/' \
              build/main_restructured.aux 2>/dev/null | head -1)
REFS_START=$(perl -ne 'print "$2\n" if /\{refs-start\}\{\{([^}]*)\}\{(\d+)\}/' \
             build/main_restructured.aux 2>/dev/null | head -1)

echo
echo "PDF:            $HERE/build/main_restructured.pdf ($PAGES pages)"
echo "Tables:         $N_BODY in body, $N_APP in appendix, $N_ALL total"
echo "Content ends:   page ${CONTENT_END:-?} (REALM limit is 8)"
echo "Refs start:     page ${REFS_START:-?}"
echo "Undefined cite: $UNDEF"
echo "TODO markers:   $(grep -c 'todo{' sections/*.tex 2>/dev/null | awk -F: '{s+=$2} END{print s+0}' || true)"
