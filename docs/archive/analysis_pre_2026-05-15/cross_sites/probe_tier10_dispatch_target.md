# Tier 10 Dispatch-Effective-Target Audit

Audit date: 2026-04-30

## Purpose
Tier 10 dispatch-effective-target mapping audit across all action types.

## Per-action mapping accuracy

| Action | Probed | Replay OK | ON_TARGET | Off-target % | Top off-target patterns |
|---|---:|---:|---:|---:|---|
| click | 18 | 18 | 1 | 94.4% | ICON_OR_IMG_INSIDE=5, BLOCK_PARENT=5, OTHER_SPAN=3 |
| type | 18 | 17 | 0 | 100.0% | OTHER=9, BLOCK_PARENT=7, ICON_OR_IMG_INSIDE=1 |
| select_option | 18 | 18 | 0 | 100.0% | OTHER=12, OTHER_SPAN=6 |

## Per-case detail

### click

- classifieds task 125 step 2 (SoM) → **ICON_OR_IMG_INSIDE** | hit `IMG.` | nearest_a=True nearest_button=False nearest_input=False
- classifieds task 8 step 22 (SoM) → **HEADING_ELEMENT** | hit `H1.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 121 step 2 (SoM) → **ICON_OR_IMG_INSIDE** | hit `IMG.` | nearest_a=True nearest_button=False nearest_input=False
- classifieds task 125 step 3 (SoM) → **ICON_OR_IMG_INSIDE** | hit `IMG.` | nearest_a=True nearest_button=False nearest_input=False
- classifieds task 223 step 1 (SoM) → **BLOCK_PARENT** | hit `P.email bld` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 66 step 3 (SoM) → **OTHER_SPAN** | hit `SPAN.currency-value` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 81 step 1 (SoM) → **ICON_OR_IMG_INSIDE** | hit `use.` | nearest_a=False nearest_button=True nearest_input=False
- reddit task 106 step 2 (SoM) → **BUTTON_LABEL_SPAN** | hit `SPAN.subscribe-button__label` | nearest_a=False nearest_button=True nearest_input=False
- reddit task 193 step 4 (SoM) → **TEXTAREA_AT_CENTER_NO_FOLLOWUP** | hit `TEXTAREA.flex__grow form-control` | nearest_a=False nearest_button=False nearest_input=True
- reddit task 84 step 1 (SoM) → **ICON_OR_IMG_INSIDE** | hit `use.` | nearest_a=False nearest_button=True nearest_input=False
- reddit task 28 step 3 (SoM) → **INPUT_AT_CENTER_AGENT_PATTERN** | hit `INPUT.form-control` | nearest_a=False nearest_button=False nearest_input=True
- reddit task 159 step 13 (SoM) → **ON_TARGET** | hit `A.submission__link` | nearest_a=True nearest_button=False nearest_input=False
- shopping task 267 step 3 (DOM) → **BLOCK_PARENT** | hit `DIV.page-wrapper` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 87 step 14 (DOM) → **BLOCK_PARENT** | hit `DIV.page-wrapper` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 165 step 3 (DOM) → **OTHER_SPAN** | hit `SPAN.base` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 407 step 6 (DOM) → **BLOCK_PARENT** | hit `LI.item` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 165 step 4 (DOM) → **OTHER_SPAN** | hit `SPAN.base` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 327 step 7 (DOM) → **BLOCK_PARENT** | hit `LI.item` | nearest_a=False nearest_button=False nearest_input=False

### type

- classifieds task 224 step 4 (SoM) → **BLOCK_PARENT** | hit `HEADER.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 232 step 6 (SoM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 159 step 9 (SoM) → **BLOCK_PARENT** | hit `HEADER.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 29 step 13 (DOM) → **BLOCK_PARENT** | hit `HEADER.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 207 step 7 (SoM) → **REPLAY_FAIL** | hit `?.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 159 step 8 (SoM) → **BLOCK_PARENT** | hit `HEADER.` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 75 step 8 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 22 step 6 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 194 step 9 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 195 step 8 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 22 step 7 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 75 step 6 (DOM) → **OTHER** | hit `NAV.site-nav` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 23 step 5 (DOM) → **ICON_OR_IMG_INSIDE** | hit `IMG.product-image-photo` | nearest_a=True nearest_button=False nearest_input=False
- shopping task 402 step 7 (DOM) → **OTHER** | hit `DD.item` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 327 step 18 (DOM) → **BLOCK_PARENT** | hit `DIV.page-title-wrapper` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 325 step 8 (DOM) → **BLOCK_PARENT** | hit `DIV.page-title-wrapper` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 387 step 18 (DOM) → **OTHER** | hit `DD.item` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 188 step 20 (DOM) → **BLOCK_PARENT** | hit `DIV.panel header` | nearest_a=False nearest_button=False nearest_input=False

### select_option

- classifieds task 156 step 11 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 122 step 10 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 122 step 14 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 154 step 5 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 133 step 16 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- classifieds task 133 step 23 (DOM) → **OTHER** | hit `LABEL.` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 186 step 1 (SoM) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 208 step 3 (SoM) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 169 step 1 (SoM) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 169 step 4 (P-prompt) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 184 step 5 (SoM) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- reddit task 169 step 3 (P-prompt) → **OTHER_SPAN** | hit `SPAN.flex__grow` | nearest_a=False nearest_button=False nearest_input=False
- shopping task 252 step 27 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
- shopping task 284 step 6 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
- shopping task 252 step 27 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
- shopping task 251 step 6 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
- shopping task 284 step 6 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
- shopping task 321 step 3 (DOM) → **OTHER** | hit `INPUT.input-text qty` | nearest_a=False nearest_button=False nearest_input=True
