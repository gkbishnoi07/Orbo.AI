"""Streamlit UI for the beauty recommender.

Presentation only. Everything about *what* to recommend lives in
`src/service.py`, so the UI and the scripted test cases cannot disagree about
what a strategy means. Nothing in this file scores, ranks or filters anything.

The layout is two columns by intent:

**Left — personalisation.** The profile panel lives in the sidebar so it stays
put while the results scroll, which is the whole reason a filter rail exists on a
retail site. It is grouped into labelled sections rather than presented as one
long stack of inputs, so it is obvious at a glance what the system wants to know.

**Right — the recommendations.** These are the point of the page, so they get the
width and nothing competes with them. Before a profile exists the column holds an
onboarding state rather than a wall of products: showing ten identical-for-
everybody products and then explaining underneath that they are not personalised
spends the most valuable screen on the least useful list.

Two rules the visual design follows from the rest of the project. Every claim on
a card is read from a column, never generated, so a card can say what a product
*fails* as readily as what it satisfies — the "Why this?" panel is grouped into
matches, evidence and caveats for exactly that reason. And prices stay in USD
because the catalogue is Sephora US; a converted figure would invent a price
nobody can pay at a rate that goes stale immediately.

One Streamlit trap worth knowing, since it already bit once: *magic* renders any
bare top-level expression as page content, so a loose docstring-style string at
module level gets printed into the UI. Use `#` comments out here.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.schema import SKIN_TONE_BANDS, SKIN_TYPES, ProductCols, Query
from src.service import (
    EVAL_ROW_FOR_STRATEGY,
    LAYER_LABELS,
    STRATEGIES,
    STRATEGIES_BY_KEY,
    RecommendationService,
    load_benchmark,
)

st.set_page_config(
    page_title="Orbo Beauty — Recommender",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preset tiers rather than a raw slider: prices run from $3 to $1,900 with a $35
# median, so a linear slider spends 95% of its travel on the long tail.
BUDGET_TIERS = [25, 50, 75, 100, 150, 250, 500, None]

# How many products the "already love" picker offers. Ranked by interactions
# actually present in the data, not by the catalogue's `review_count` — see
# `RecommendationService.history_pool`. Every option can switch collaborative
# filtering on; ranking by `review_count` produced a picker where most could not.
HISTORY_POOL = 2000

FIND_LABEL = "Find my products"

# Every profile widget is keyed so "Clear filters" can actually empty it.
# Without keys Streamlit owns the state anonymously and there is nothing to pop,
# which is why a reset button would appear to work and change nothing.
PROFILE_KEYS = (
    "f_skin_type",
    "f_skin_tone",
    "f_concerns",
    "f_category",
    "f_budget",
    "f_owned",
)


# Paging. A page is 20 and the list grows to MAX_SHOWN on demand.
#
# There is a ceiling, and it is worth being honest about why rather than
# pretending "all 5,694" is a thing anyone wants. Each card is roughly eight
# Streamlit elements, so the full eligible set would be ~45,000 elements — the
# page would take minutes to paint and be unusable when it did. Beyond that,
# ranked position 3,000 of a recommendation list carries no information: the
# whole point of ranking is that the top matters. When the ceiling is reached the
# UI says so and points at the filters, which is the useful way to narrow 5,694
# products down.
#
# Comments, not docstring-style strings: Streamlit's magic renders any bare
# top-level expression as page content.
PAGE_SIZE = 20
MAX_SHOWN = 100


# Injected once, only on the render immediately after a submit. Streamlit has no
# Python API for collapsing the sidebar, and on a phone the open panel covers the
# results the user just asked for — they press "Find my products" and appear to
# get nothing. Clicking Streamlit's own collapse control is the only route; it
# carries a stable `data-testid`, and the width check keeps desktop untouched
# where the sidebar costs nothing.
COLLAPSE_SIDEBAR_ON_MOBILE = """
<script>
(function () {
  const doc = window.parent && window.parent.document;
  if (!doc) return;
  const NARROW = 768;

  function collapse(attempt) {
    if (window.parent.innerWidth > NARROW) return;
    const sidebar = doc.querySelector('[data-testid="stSidebar"]');
    // aria-expanded is the only reliable signal that it is still open; without
    // it a retry would keep re-clicking and toggle the panel back open.
    if (sidebar && sidebar.getAttribute("aria-expanded") === "false") return;
    const button = doc.querySelector('[data-testid="stSidebarCollapseButton"] button')
      || doc.querySelector('[data-testid="stSidebarCollapseButton"]');
    if (button) { button.click(); return; }
    // The control mounts a beat after the rerun paints.
    if (attempt < 12) setTimeout(() => collapse(attempt + 1), 120);
  }
  collapse(0);
})();
</script>
"""


def show_more() -> None:
    current = st.session_state.get("shown", PAGE_SIZE)
    st.session_state.shown = min(current + PAGE_SIZE, MAX_SHOWN)


def clear_filters() -> None:
    """Reset the profile and return to onboarding.

    Runs as a button callback rather than inline: callbacks fire before the
    script reruns, so popping widget keys here takes effect on the next render
    instead of colliding with widgets that have already been instantiated.
    """
    for key in PROFILE_KEYS:
        st.session_state.pop(key, None)
    st.session_state.mode = None
    st.session_state.shown = PAGE_SIZE

# Placeholder plates are tinted by primary category, not by a hash of the product
# id. The dataset ships no imagery and inventing some would be dishonest, but a
# random colour per product makes the grid look like noise; a stable tint per
# category makes it look organised and carries a little information.
#
# The colours themselves live in CSS rather than here, because an inline
# background cannot respond to dark mode and these pastels glow on a dark ground.
CATEGORY_SLUGS = {
    "Skincare": "skincare",
    "Makeup": "makeup",
    "Hair": "hair",
    "Fragrance": "fragrance",
    "Bath & Body": "bath",
    "Men": "men",
    "Tools & Brushes": "tools",
    "Mini Size": "mini",
    "Gifts": "gifts",
}

STEPS = [
    ("01", "Describe your skin", "Type, tone, and what you would like to improve."),
    (
        "02",
        "We match products",
        "Against ingredients and labels, shoppers like you, and overall popularity.",
    ),
    (
        "03",
        "See why each pick fits",
        "Every card shows its reasoning, including what it does not satisfy.",
    ),
]

# Injected once per run into a zero-height iframe, which `components.html`
# serves same-origin so the script can reach `window.parent.document` and do two
# things Python cannot.
#
# `components.html` carries a deprecation notice pointing at `st.iframe`, but
# `st.iframe` takes a URL rather than markup, and a `data:` URL would load into
# an opaque origin with no access to the parent document — which is the entire
# mechanism here. So this stays until there is a same-origin inline replacement.
#
# 1. Hide the Print and Record screen entries. Those are browser features rather
#    than product features, and they share Streamlit's viewer menu with the
#    light/dark control — `toolbarMode` can only remove the whole menu, taking
#    the theme control with it. The entries carry a stable
#    `data-testid="stMainMenuItemLabel"`, so matching on their text is the only
#    way to drop exactly those two.
#
# 2. Stamp `data-appearance` on the root element by measuring the background
#    Streamlit actually painted. The custom CSS below keys off that instead of
#    `prefers-color-scheme`, which reports the *operating system* and so
#    disagreed with an explicit in-app Light choice — the cause of grey text on
#    white in an earlier revision.
#
# Both run under a MutationObserver because the menu popover is built on demand
# and Streamlit repaints on every rerun. If a future Streamlit renames the
# testid, the entries simply reappear and the appearance falls back to light —
# it degrades, it does not break.
THEME_SYNC = """
<script>
(function () {
  const doc = window.parent && window.parent.document;
  if (!doc) return;
  const HIDDEN = new Set(["Print", "Record screen", "Stop recording"]);

  function hideBrowserFeatures() {
    doc.querySelectorAll('[data-testid="stMainMenuItemLabel"]').forEach((label) => {
      if (!HIDDEN.has((label.textContent || "").trim())) return;
      const row = label.closest('li,[role="menuitem"],[role="option"]') || label.parentElement;
      if (row) row.style.display = "none";
    });
  }

  // Walk up for the first ancestor that actually paints something. A
  // transparent element computes to rgba(0, 0, 0, 0), and reading those first
  // three numbers as a colour says "black" — which stamped dark onto a light
  // page and left grey text on white.
  function paintedBackground(el) {
    while (el) {
      const parts = (getComputedStyle(el).backgroundColor || "").match(/[\\d.]+/g);
      if (parts && parts.length >= 3) {
        const alpha = parts.length > 3 ? Number(parts[3]) : 1;
        if (alpha > 0.05) return parts.slice(0, 3).map(Number);
      }
      el = el.parentElement;
    }
    return null;
  }

  function syncAppearance() {
    const start = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    const rgb = paintedBackground(start) || paintedBackground(doc.body);
    // No opaque ancestor found: leave the attribute alone so the CSS keeps its
    // light default rather than guessing.
    if (!rgb) return;
    const [r, g, b] = rgb;
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    doc.documentElement.setAttribute("data-appearance", luminance < 0.5 ? "dark" : "light");
  }

  function apply() { hideBrowserFeatures(); syncAppearance(); }
  apply();
  // Attributes and style, not just childList: switching theme in Streamlit's
  // menu repaints via style changes and inserts no nodes, so a childList-only
  // observer never re-measured and the palette stayed latched on whatever the
  // first paint happened to be — light text left on a white page.
  new MutationObserver(apply).observe(doc.documentElement, {
    childList: true, subtree: true, attributes: true,
    attributeFilter: ["style", "class", "data-theme"],
  });
  // Backstop: the repaint can land in a frame the observer does not see.
  setInterval(apply, 400);
})();
</script>
"""

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&display=swap');

  /* Font family is set through .streamlit/config.toml, not here. Forcing
     font-family onto every div and span also hits Streamlit's Material icon
     spans, and an icon font asked to render "arrow_right" prints the ligature
     name as literal text on top of the label. */
  [data-testid="stIconMaterial"], .material-icons, .material-symbols-rounded {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
  }

  /* Tokens. Both blocks mirror the palettes in config.toml; with `base` unset
     Streamlit follows the system preference too, so the two stay in step. */
  /* Tokens mirror config.toml and switch on [data-appearance], which THEME_SYNC
     sets from the background Streamlit actually painted. Deliberately NOT
     prefers-color-scheme: that reports the operating system, so it disagreed
     with an explicit in-app Light choice and put grey text on white.
     `--muted` and `--faint` are darker than a typical grey ramp because at these
     sizes anything lighter fails contrast. */
  :root, :root[data-appearance="light"] {
    --line: #DED3CB;     --surface: #FFFFFF; --panel: #F7F2EE;
    --accent: #A32E4E;   --accent-bg: #F7E8ED; --accent-line: #E4CBD4;
    --chip-bg: #F1E9E3;  --ok: #256B47;      --star: #A87F1E;
  }
  :root[data-appearance="dark"] {
    --line: #3A323E;     --surface: #1E1A22; --panel: #221D26;
    --accent: #E68DA3;   --accent-bg: #34222B; --accent-line: #4D323D;
    --chip-bg: #272029;  --ok: #74C598;      --star: #DCBA5F;
  }

  /* Text colour is INHERITED from Streamlit, never from a token of our own.
     Streamlit always sets the body colour correctly for the active theme, so
     inheriting makes the old failure impossible: if the appearance probe were
     ever wrong or slow, the worst case is a slightly-off surface tint, never
     unreadable text. Muted tones are opacity on the inherited colour, which
     stays legible whichever way round the theme is. */
  .stApp .pname, .stApp .fact b, .stApp .hero h1, .stApp .onboard h2,
  .stApp .stepcol b, .stApp .mark b, .stApp .pmeta b { color: inherit; }

  .stApp .hero p, .stApp .onboard p, .stApp .stepcol span, .stApp .pmeta,
  .stApp .chip, .stApp .ev, .stApp .barlabel { color: inherit; opacity: .78; }

  .stApp .marksub, .stApp .grouplabel, .stApp .fact span, .stApp .detail,
  .stApp .meta, .stApp .plate .cat, .stApp .whyhead,
  .stApp .pmeta u { color: inherit; opacity: .58; }

  .stApp .meta b { opacity: 1; }
  .stApp .ev.no { opacity: .55; }

  .block-container { padding-top: 2rem; padding-bottom: 4rem; max-width: 1380px; }

  /* Streamlit appends an anchor link to every heading it renders. On the hero it
     shows up as a chain icon floating in the middle of the title. */
  .hero a[href^="#"], .onboard a[href^="#"] { display: none !important; }

  /* ---------- sidebar ---------- */
  .mark { display: flex; align-items: baseline; gap: .45rem;
          padding-bottom: .75rem; border-bottom: 1px solid var(--line); }
  .mark b { font-size: .8rem; font-weight: 700; letter-spacing: .16em;
            text-transform: uppercase; color: var(--ink); }
  .mark i { font-style: normal; color: var(--accent); font-size: .95rem; }
  .marksub { font-size: .7rem; color: var(--faint); margin: .45rem 0 .2rem; }

  .grouplabel { font-size: .63rem; font-weight: 700; letter-spacing: .13em;
                text-transform: uppercase; color: var(--faint);
                margin: 1.5rem 0 .2rem; }

  /* ---------- hero ---------- */
  .hero h1 { font-size: 2.7rem; font-weight: 400; letter-spacing: -.01em;
             line-height: 1.1; margin: 0 0 .45rem; color: var(--ink); }
  .hero h1 em { font-style: italic; color: var(--accent); }
  .hero p { color: var(--muted); margin: 0; font-size: 1rem; max-width: 56ch;
            line-height: 1.5; }
  .rule { border-bottom: 1px solid var(--line); margin: 1.6rem 0 1.4rem; }

  /* ---------- onboarding ---------- */
  .onboard { background: var(--panel); border: 1px solid var(--line);
             border-radius: 14px; padding: 2rem 2.2rem; }
  .onboard h2 { font-size: 1.55rem; font-weight: 400; margin: 0 0 .45rem;
                color: var(--ink); }
  .onboard p { color: var(--muted); margin: 0; font-size: .93rem; max-width: 54ch;
               line-height: 1.55; }
  .steps { display: flex; gap: 2.2rem; flex-wrap: wrap; margin-top: 1.6rem; }
  .stepcol { flex: 1 1 185px; min-width: 170px; }
  .stepno { font-size: 1.3rem; color: var(--accent); opacity: .55; line-height: 1; }
  .stepcol b { display: block; font-size: .86rem; color: var(--ink);
               font-weight: 600; margin: .4rem 0 .16rem; }
  .stepcol span { font-size: .79rem; color: var(--muted); line-height: 1.5; }

  .facts { display: flex; gap: 2.6rem; flex-wrap: wrap; }
  .fact b { display: block; font-size: 1.1rem; color: var(--ink); font-weight: 600;
            font-variant-numeric: tabular-nums; }
  .fact span { font-size: .64rem; text-transform: uppercase; letter-spacing: .1em;
               color: var(--faint); }

  /* ---------- results header ---------- */
  .summary { display: flex; align-items: center; flex-wrap: wrap; gap: .5rem;
             margin-bottom: .5rem; }
  .chip { background: var(--chip-bg); border: 1px solid var(--line);
          border-radius: 999px; padding: .2rem .66rem; font-size: .755rem;
          color: var(--muted); white-space: nowrap; }
  .chip.key { background: var(--accent-bg); border-color: var(--accent-line);
              color: var(--accent); font-weight: 600; }
  .meta { font-size: .795rem; color: var(--faint);
          font-variant-numeric: tabular-nums; }
  .meta b { color: var(--muted); font-weight: 600; }

  /* ---------- product cards ---------- */
  /* Stretch so every card in a row shares the tallest height. */
  div[data-testid="stHorizontalBlock"] { align-items: stretch; }
  div[data-testid="stColumn"] > div,
  div[data-testid="stColumn"] div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100%;
  }
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.pface) {
    border: 1px solid var(--line) !important; border-radius: 12px !important;
    background: var(--surface); overflow: hidden;
    display: flex; flex-direction: column;
  }

  /* Every card is built from fixed-height parts, so a row is uniform without
     relying on flex stretch reaching through Streamlit's wrapper divs — those
     testids change between versions and the alignment quietly broke when they
     did. Fixed parts align whether or not the stretch rules match. */
  .pface { display: flex; flex-direction: column; }
  .plate { margin: -1rem -1rem .7rem; height: 92px; flex: 0 0 92px;
           display: flex; flex-direction: column; align-items: center;
           justify-content: center; gap: .25rem; position: relative;
           padding: 0 .9rem; }
  .plate .wm { font-family: 'Instrument Serif', Georgia, serif; font-size: 1rem;
               line-height: 1.14; text-align: center; width: 100%;
               display: -webkit-box; -webkit-line-clamp: 2;
               -webkit-box-orient: vertical; overflow: hidden; }
  .plate .cat { font-size: .555rem; letter-spacing: .13em; text-transform: uppercase;
                opacity: .68; white-space: nowrap; overflow: hidden;
                text-overflow: ellipsis; max-width: 100%; }
  .rank { position: absolute; top: .45rem; left: .5rem;
          background: rgba(255,255,255,.82); color: #241F22; font-size: .63rem;
          font-weight: 700; padding: .08rem .34rem; border-radius: 4px;
          font-variant-numeric: tabular-nums; }

  .brand { font-size: .63rem; font-weight: 700; text-transform: uppercase;
           letter-spacing: .1em; color: var(--accent); white-space: nowrap;
           overflow: hidden; text-overflow: ellipsis; height: 1.15em;
           line-height: 1.15em; }
  .pname { font-size: .85rem; line-height: 1.3; color: var(--ink);
           font-weight: 500; margin: .15rem 0 .35rem; height: 2.6em;
           display: -webkit-box; -webkit-line-clamp: 2;
           -webkit-box-orient: vertical; overflow: hidden; }
  .pmeta { font-size: .76rem; white-space: nowrap; height: 1.5em;
           line-height: 1.5em; font-variant-numeric: tabular-nums;
           margin-bottom: .4rem; overflow: hidden; text-overflow: ellipsis; }
  .pmeta b { font-size: .95rem; font-weight: 700; color: var(--ink); }
  .pmeta i { font-style: normal; color: var(--star); }
  .pmeta u { text-decoration: none; color: var(--faint); }

  /* Exactly two rows, each clamped to one line. A wrapped third line was what
     made neighbouring cards different heights; the full text is one click away
     in "Why this?", so truncating here costs nothing. */
  .evbox { height: 2.9em; overflow: hidden; }
  .ev { font-size: .74rem; line-height: 1.45; margin: 0;
        display: flex; gap: .3rem; align-items: baseline;
        white-space: nowrap; overflow: hidden; }
  .ev .g { flex: 0 0 auto; }
  .ev > span:last-child { overflow: hidden; text-overflow: ellipsis;
                          white-space: nowrap; min-width: 0; }
  /* The expander must sit at the same height on every card in the row. */
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.pface) details,
  div[data-testid="stVerticalBlockBorderWrapper"]:has(.pface) [data-testid="stExpander"] {
    margin-top: auto;
  }
  .yes { color: var(--muted); } .yes .g { color: var(--ok); }
  .no  { color: var(--faint); } .no  .g { color: var(--faint); }
  .detail { color: var(--faint); font-size: .69rem; }

  .whyhead { font-size: .62rem; font-weight: 700; letter-spacing: .11em;
             text-transform: uppercase; color: var(--faint); margin: .7rem 0 .25rem; }

  .barlabel { font-size: .69rem; color: var(--muted); display: flex;
              justify-content: space-between; }
  .barwrap { background: var(--chip-bg); border-radius: 3px; height: 5px;
             margin: .15rem 0 .45rem; }
  .bar { background: var(--accent); height: 5px; border-radius: 3px; }

  /* Category plates. Light pastels invert to deep, desaturated grounds rather
     than staying pale, which would glare against a dark canvas. */
  .plate.t-skincare  { background:#EFE4E6; color:#8C4257; }
  .plate.t-makeup    { background:#EDE3EA; color:#7A4463; }
  .plate.t-hair      { background:#E4E9E4; color:#41604A; }
  .plate.t-fragrance { background:#F0E8DC; color:#7C5A32; }
  .plate.t-bath      { background:#E3E8EC; color:#3F5A6B; }
  .plate.t-men       { background:#E6E5E9; color:#4B4A5C; }
  .plate.t-tools     { background:#EBE7E2; color:#6A6058; }
  .plate.t-mini      { background:#F0E9E4; color:#7A6252; }
  .plate.t-gifts     { background:#F1E5E8; color:#8A4A5E; }
  .plate.t-other     { background:#ECE7E3; color:#6A6167; }

  /* Light pastels become deep desaturated grounds rather than staying pale,
     which would glare against a dark canvas. */
  :root[data-appearance="dark"] .plate.t-skincare  { background:#31232A; color:#E3ADBB; }
  :root[data-appearance="dark"] .plate.t-makeup    { background:#2E2334; color:#D9AECE; }
  :root[data-appearance="dark"] .plate.t-hair      { background:#232E27; color:#A6CBB1; }
  :root[data-appearance="dark"] .plate.t-fragrance { background:#332A20; color:#D9BC8E; }
  :root[data-appearance="dark"] .plate.t-bath      { background:#212B31; color:#A3C2D1; }
  :root[data-appearance="dark"] .plate.t-men       { background:#28262E; color:#B7B4C4; }
  :root[data-appearance="dark"] .plate.t-tools     { background:#2B2823; color:#C6BCAE; }
  :root[data-appearance="dark"] .plate.t-mini      { background:#2E2721; color:#CDB6A2; }
  :root[data-appearance="dark"] .plate.t-gifts     { background:#31232B; color:#DFAABC; }
  :root[data-appearance="dark"] .plate.t-other     { background:#282429; color:#B4AAB2; }
  :root[data-appearance="dark"] .rank { background: rgba(16,13,18,.75); color:#F3EDEF; }

  /* The injector is a zero-height iframe with no visible content. */
  div[data-testid="stIFrame"]:has(+ *) , iframe[title="st.iframe"] { display: none; }

  @media (max-width: 1400px) { .block-container { max-width: 100%; } }
  @media (max-width: 1200px) { .hero h1 { font-size: 2.25rem; } }
</style>
"""


def tint_class(primary_category: str | None) -> str:
    return "t-" + CATEGORY_SLUGS.get(str(primary_category), "other")


@st.cache_resource(show_spinner="Loading catalogue and fitting models…")
def load_service() -> RecommendationService:
    return RecommendationService.from_artifacts()


@st.cache_data(show_spinner=False)
def history_options(_key: str) -> list[tuple[str, str]]:
    """(product_id, label) for products collaborative filtering can actually use."""
    return load_service().history_pool(HISTORY_POOL)


def compact_reviews(count: int) -> str:
    if count >= 1000:
        return f"{count / 1000:.1f}k".replace(".0k", "k")
    return str(count)


def render_sidebar(service: RecommendationService) -> dict:
    """The personalisation panel. Grouped so it is scannable, not a wall of inputs."""
    skin_concerns = service.concerns_for("skin")
    hair_concerns = service.concerns_for("hair")
    concern_labels = dict(skin_concerns + hair_concerns)
    options = history_options("v1")
    owned_labels = dict(options)
    pairs = service.categories()
    primary_of = {secondary: primary for primary, secondary in pairs}

    with st.sidebar:
        st.markdown(
            '<div class="mark"><i>✦</i><b>Orbo Beauty</b></div>'
            '<div class="marksub">Skin-first product recommendations</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="grouplabel">Your skin</div>', unsafe_allow_html=True)
        skin_type = st.radio(
            "Skin type",
            SKIN_TYPES,
            index=None,
            horizontal=True,
            format_func=str.capitalize,
            help="Decides which group of reviewers you are compared against.",
            key="f_skin_type",
        )
        skin_tone = st.selectbox(
            "Skin tone",
            SKIN_TONE_BANDS,
            index=None,
            placeholder="Select your skin tone",
            format_func=str.capitalize,
            help="Five bands, grouped so each holds enough reviewers to mean "
            "something.",
            key="f_skin_tone",
        )

        st.markdown(
            '<div class="grouplabel">What you want to improve</div>',
            unsafe_allow_html=True,
        )
        chosen = st.pills(
            "Concerns",
            options=list(concern_labels),
            selection_mode="multi",
            format_func=lambda slug: concern_labels[slug],
            label_visibility="collapsed",
            help="Choosing Sensitivity also removes anything containing fragrance, "
            "drying alcohol or harsh sulfates.",
            key="f_concerns",
        )

        st.markdown(
            '<div class="grouplabel">Narrow it down</div>', unsafe_allow_html=True
        )
        category = st.selectbox(
            "Category",
            list(primary_of),
            index=None,
            placeholder="All categories",
            format_func=lambda value: f"{primary_of[value]} › {value}",
            key="f_category",
        )
        budget = st.select_slider(
            "Budget",
            options=BUDGET_TIERS,
            value=100,
            format_func=lambda v: "No limit" if v is None else f"Under ${v}",
            key="f_budget",
        )
        owned = st.multiselect(
            "Products you already love",
            options=[pid for pid, _ in options],
            format_func=owned_labels.get,
            placeholder="Optional — search a brand",
            help="The strongest signal there is. Lets us find what people who liked "
            "the same things went on to rate highly.",
            key="f_owned",
        )

        has_profile = bool(skin_type or skin_tone or chosen or owned)
        st.write("")
        submitted = st.button(
            f"✦  {FIND_LABEL}", type="primary", width="stretch", disabled=not has_profile
        )
        # A radio cannot be unselected by clicking it again, so without this there
        # is no way back to "no skin type" once one is chosen — the reset lives
        # beside the inputs it resets, not only up in the results header.
        st.button(
            "Clear all",
            width="stretch",
            on_click=clear_filters,
            disabled=not has_profile,
            help="Empties every field and returns to the start.",
            key="sidebar_clear",
        )
        if not has_profile:
            st.caption("Pick a skin type or a concern to continue.")

        st.markdown(
            '<div class="grouplabel">Your preferences</div>', unsafe_allow_html=True
        )
        strategy = st.selectbox(
            "Approach",
            [s.key for s in STRATEGIES],
            format_func=lambda key: STRATEGIES_BY_KEY[key].label,
        )
        st.caption(STRATEGIES_BY_KEY[strategy].blurb)

    return {
        "skin_type": skin_type,
        "skin_tone": skin_tone,
        "concerns": list(chosen or []),
        "concern_labels": concern_labels,
        "category": category,
        "budget": budget,
        "owned": list(owned),
        "has_profile": has_profile,
        "submitted": submitted,
        "strategy": strategy,
    }


def render_hero() -> None:
    st.markdown(
        '<div class="hero"><h1>Find products that fit <em>your</em> skin</h1>'
        "<p>Personalised recommendations backed by real evidence — 1.09 million "
        "ratings from people who told us their skin type.</p></div>",
        unsafe_allow_html=True,
    )


def render_onboarding(service: RecommendationService) -> None:
    steps = "".join(
        f'<div class="stepcol"><div class="stepno">{number}</div>'
        f"<b>{title}</b><span>{body}</span></div>"
        for number, title, body in STEPS
    )
    st.markdown(
        '<div class="onboard"><h2>Tell us about your skin</h2>'
        "<p>Choose your skin type and what you would like to improve. We will use "
        "those signals to find products that fit you.</p>"
        f'<div class="steps">{steps}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

    facts = service.facts
    st.markdown(
        '<div class="facts">'
        f'<div class="fact"><b>{facts.n_products:,}</b><span>Products</span></div>'
        f'<div class="fact"><b>{facts.n_brands}</b><span>Brands</span></div>'
        f'<div class="fact"><b>{facts.n_interactions / 1e6:.2f}M</b>'
        "<span>Real ratings</span></div>"
        f'<div class="fact"><b>{facts.reviewed_share:.0%}</b>'
        "<span>Catalogue rated</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_summary(profile: dict, personalised: bool) -> None:
    """The compact profile strip shown above the results."""
    chips: list[tuple[str, bool]] = []
    if personalised:
        if profile["skin_type"]:
            chips.append((f"{profile['skin_type'].capitalize()} skin", True))
        if profile["skin_tone"]:
            chips.append((f"{profile['skin_tone'].capitalize()} tone", False))
        for slug in profile["concerns"]:
            chips.append((profile["concern_labels"][slug], False))
        if profile["owned"]:
            count = len(profile["owned"])
            chips.append((f"{count} product{'s' if count != 1 else ''} you love", False))
    else:
        chips.append(("Not personalised", True))
    if profile["category"]:
        chips.append((profile["category"], False))
    chips.append(
        ("Any price" if profile["budget"] is None else f"Under ${profile['budget']}", False)
    )

    rendered = "".join(
        f'<span class="chip{" key" if key else ""}">{text}</span>' for text, key in chips
    )
    st.markdown(f'<div class="summary">{rendered}</div>', unsafe_allow_html=True)


def render_card(service: RecommendationService, scored, rank: int) -> None:
    product = service.product(scored.product_id)
    if product is None:
        return

    brand = str(product[ProductCols.BRAND])
    rating = product.get(ProductCols.RATING)
    # 278 products (3.3%) carry no review count at all. `x or 0` does not help
    # here: NaN is truthy, so it passes straight through and int() raises, taking
    # the whole page down. Only strategies that can surface an unreviewed product
    # — random, content-only — ever hit it, which is why it stayed hidden.
    raw_reviews = product.get(ProductCols.N_REVIEWS)
    reviews = 0 if pd.isna(raw_reviews) else int(raw_reviews)
    bits = [f"<b>${float(product[ProductCols.PRICE]):,.0f}</b>"]
    if not pd.isna(rating):
        bits.append(f"<i>★</i> {float(rating):.1f}")
    if reviews:
        bits.append(f"<u>{compact_reviews(reviews)}</u>")

    # Matches first, then two lines only: the card stays short and the full
    # picture — caveats included — is one click away in the expander.
    inline = sorted(scored.evidence, key=lambda e: not e.supported)[:2]
    lines = "".join(
        f'<div class="ev {"yes" if e.supported else "no"}">'
        f'<span class="g">{"✓" if e.supported else "○"}</span>'
        f"<span>{e.label}</span></div>"
        for e in inline
    )

    with st.container(border=True):
        # A single block, not one per element: Streamlit inserts a gap between
        # elements, so a per-line render made otherwise identical cards different
        # heights depending on how many evidence lines wrapped.
        st.markdown(
            f'<div class="pface">'
            f'<div class="plate {tint_class(product.get(ProductCols.PRIMARY_CATEGORY))}">'
            f'<div class="rank">{rank}</div>'
            f'<div class="wm">{brand}</div>'
            f'<div class="cat">{product.get(ProductCols.CATEGORY, "")}</div></div>'
            f'<div class="brand">{brand}</div>'
            f'<div class="pname">{product[ProductCols.NAME]}</div>'
            f'<div class="pmeta">{" · ".join(bits)}</div>'
            f'<div class="evbox">{lines}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

        with st.expander("Why this?"):
            groups = (
                ("Why it matches", "match"),
                ("Evidence", "evidence"),
                ("Worth knowing", "caveat"),
            )
            for title, kind in groups:
                items = [e for e in scored.evidence if e.kind == kind]
                if not items:
                    continue
                body = "".join(
                    f'<div class="ev {"yes" if e.supported else "no"}">'
                    f'<span class="g">{"✓" if e.supported else "○"}</span>'
                    f"<span>{e.label}"
                    + (f'<br><span class="detail">{e.detail}</span>' if e.detail else "")
                    + "</span></div>"
                    for e in items
                )
                st.markdown(
                    f'<div class="whyhead">{title}</div>{body}', unsafe_allow_html=True
                )

            shown = {k: v for k, v in scored.components.items() if k in LAYER_LABELS}
            if shown:
                bars = "".join(
                    f'<div class="barlabel"><span>{label}</span>'
                    f"<span>{max(0.0, min(1.0, shown[key])):.2f}</span></div>"
                    f'<div class="barwrap"><div class="bar" '
                    f'style="width:{max(0.0, min(1.0, shown[key])) * 100:.0f}%"></div></div>'
                    for key, label in LAYER_LABELS.items()
                    if key in shown
                )
                st.markdown(
                    f'<div class="whyhead">What drove this score</div>{bars}',
                    unsafe_allow_html=True,
                )
            st.caption(f"`{scored.product_id}` · final score {scored.score:.3f}")


def render_about(service: RecommendationService, query: Query) -> None:
    """Methodology and honest numbers, kept out of the shopping path."""
    facts = service.facts

    with st.expander("How this works, and how well it does"):
        product_tab, technical_tab, performance_tab = st.tabs(
            ["The three signals", "Technical detail", "Model performance"]
        )

        with product_tab:
            st.markdown(
                """
**Product match.** Reads each product's own ingredients, labels and category and
compares them with what you asked for. This is the only signal that works for a
product nobody has rated yet — and that is most of the shelf.

**Similar shoppers.** Finds products rated highly by people with a skin profile
like yours who liked what you like. This catches things product copy never says:
which foundation oxidises, which moisturiser pills under sunscreen.

**Overall popularity.** Breaks ties toward things lots of people rate well.

**Your skin tone.** A small nudge toward products that people in your tone band
rate more highly than everyone else does. Deliberately small, and it only moves
a product when that band's opinion actually differs from the average — nothing is
ever excluded for lacking tone data.

The four are weighed together and the final list is diversified slightly, so you
do not get ten near-identical serums.

**The reasons are real.** Every line under "Why this?" is read from the data, not
written by an AI, so it cannot invent a claim about a product. When we say 86% of
dry-skin reviewers rated something 4+, the number of reviewers is shown beside
it — and if fewer than 30 people in your group reviewed it, we do not quote a
percentage at all.

**Prices are the catalogue's own, in US dollars.** This is the Sephora US
catalogue; converting would invent a price at a rate that goes stale immediately.
"""
            )

        with technical_tab:
            st.markdown(
                f"""
Hard filters run first — budget, category, and suitability rules no score can
override. Then three scoring layers, min-max scaled and blended, then MMR:

1. **Content** — cosine similarity over precomputed TF-IDF vectors (256 dims,
   SVD-reduced) of the product text blob: brand, name, categories, highlights and
   ingredients. Plus exact concern and skin-type tag matching. Covers the
   **{1 - facts.reviewed_share:.0%}** of the catalogue with no interactions.
2. **Cohort collaborative** — item-item cosine similarity with co-occurrence
   shrinkage, rebuilt per skin-type cohort. Your cohort holds
   **{service.cohort_size(query.skin_type):,}** interactions.
3. **Popularity prior** — log-damped count of ratings at 4+.

TF-IDF is used rather than a sentence transformer because it measured better
here: the product text is highlight tokens and INCI ingredient lists, not prose,
so term overlap beats semantic similarity. The exact ratio is in the Model
performance tab, read from the recorded run rather than repeated here.

Vectors are precomputed offline, so the deployed app never loads torch. Two
different latencies get quoted for systems like this and they are not the same
number: **model inference** (`Recommender.recommend()`, what the evaluation
harness times) and **end-to-end recommendation** (inference plus explanation
generation, before any Streamlit rendering). The strip above every result page
reports the second one, measured live for that request.

**Untagged is not unsuitable.** Only 12.4% of products carry a skin-type tag, so
the rules fire on stated disagreement, never on a missing tag. Treating untagged
as unsuitable would silently delete seven eighths of the catalogue.
"""
            )

        with performance_tab:
            benchmark = load_benchmark()
            if benchmark is None:
                st.warning(
                    "No recorded evaluation found. Run "
                    "`python scripts/04_evaluate.py` to generate "
                    "`reports/evaluation.json`."
                )
            else:
                k = benchmark.get("k", 10)
                rows = []
                for info in STRATEGIES:
                    row = benchmark["models"].get(EVAL_ROW_FOR_STRATEGY[info.key])
                    if not row:
                        continue
                    warm, cold = row["warm"], row["cold"] or {}
                    rows.append(
                        {
                            "Approach": info.label,
                            "Returning user": f"{warm.get(f'ndcg@{k}', float('nan')):.4f}",
                            "New user": f"{cold.get(f'ndcg@{k}', float('nan')):.4f}",
                            "Catalogue coverage": f"{warm.get('coverage', 0):.1%}",
                        }
                    )
                st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
                st.caption(
                    f"**Recorded benchmark**, not measured live — NDCG@{k}, "
                    f"leave-one-out on {benchmark.get('warm_cases', 0):,} warm and "
                    f"{benchmark.get('cold_cases', 0):,} cold held-out likes, from "
                    f"commit `{benchmark.get('commit', 'unknown')}` "
                    f"(`{benchmark.get('embedding_method', '?')}` vectors). Every "
                    "model is fitted on the same training split with the evaluated "
                    "interactions removed. Regenerate with "
                    "`python scripts/04_evaluate.py`."
                )
            if benchmark is not None:
                k = benchmark.get("k", 10)
                def ndcg(row_key: str, regime: str) -> float:
                    row = benchmark["models"].get(row_key) or {}
                    part = row.get(regime) or {}
                    return float(part.get(f"ndcg@{k}", float("nan")))

                unrated = facts.n_products - facts.n_reviewed_products
                st.markdown(
                    f"""
**Read these honestly.**

- *Similar shoppers* alone scores {ndcg('cf-only','warm'):.4f} on a returning
  user against the blend's {ndcg('hybrid-tfidf','warm'):.4f}. The blend is kept
  because held-out items always come from the ratings table, so the
  {unrated:,} unrated products can never count as a hit and the product-match
  layer's whole contribution is invisible to this metric.
- For a brand-new user the blend scores {ndcg('hybrid-tfidf','cold'):.4f} against
  {ndcg('popularity','cold'):.4f} for plain popularity. Cold start is genuinely
  hard and we do not claim otherwise.
- *Random* has the highest catalogue coverage, which is why coverage is never
  read on its own.
- Latency in the recorded run is **model inference only**. It times
  `Recommender.recommend()` and excludes explanation generation and all
  Streamlit rendering, so it is not end-to-end user latency.

Full numbers in `reports/evaluation.md`; the blend sweep, including the weights
that were rejected, in `reports/weight_sweep.md`.
"""
                )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    components.html(THEME_SYNC, height=0)

    if not Path("artifacts/products.parquet").exists():
        st.error(
            "Precomputed artifacts are missing. Run `python scripts/03_build_artifacts.py`, "
            "or see the README for the full reproduction steps."
        )
        st.stop()

    service = load_service()
    facts = service.facts
    st.session_state.setdefault("mode", None)
    st.session_state.setdefault("shown", PAGE_SIZE)

    profile = render_sidebar(service)
    if profile["submitted"]:
        st.session_state.mode = "personalised"
        st.session_state.shown = PAGE_SIZE  # a new profile starts a new list
        st.session_state.collapse_sidebar = True
        st.rerun()

    # One-shot: consumed here so it fires on this render only, never on the
    # reruns that follow when a filter is nudged.
    if st.session_state.pop("collapse_sidebar", False):
        components.html(COLLAPSE_SIDEBAR_ON_MOBILE, height=0)

    render_hero()

    if st.session_state.mode is None:
        st.write("")
        render_onboarding(service)
        # A quiet way past personalisation. Kept deliberately secondary: it is a
        # useful baseline to compare against, but it is not what the page is for.
        _, escape = st.columns([2, 1])
        with escape:
            if st.button(
                "Browse popular products instead",
                width="stretch",
                help="Skips personalisation and ranks by what everyone rates highly.",
            ):
                st.session_state.mode = "popular"
                st.session_state.collapse_sidebar = True
                st.rerun()
        st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
        render_about(service, Query())
        return

    personalised = st.session_state.mode == "personalised" and profile["has_profile"]

    # In popular mode the profile fields are deliberately left out of the query,
    # not merely unmentioned. Category and budget are constraints and still
    # apply, but a list labelled "not personalised" must not quietly be ranked
    # using someone's skin type.
    query = Query(
        skin_type=profile["skin_type"] if personalised else None,
        skin_tone=profile["skin_tone"] if personalised else None,
        concerns=tuple(profile["concerns"]) if personalised else (),
        category=profile["category"],
        budget_max=float(profile["budget"]) if profile["budget"] is not None else None,
        liked_product_ids=tuple(profile["owned"]) if personalised else (),
    )
    # A list headed "Popular right now" has to actually be ranked by popularity.
    # Reading the sidebar's Approach here meant picking Random and pressing
    # "Browse popular products" produced random items under that heading — the
    # label describing something the ranking was not doing.
    strategy = profile["strategy"] if personalised else "popularity"
    info = STRATEGIES_BY_KEY[strategy]

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    summary_col, clear_col = st.columns([5, 1], vertical_alignment="center")
    with summary_col:
        render_summary(profile, personalised)
    with clear_col:
        st.button(
            "Clear filters",
            width="stretch",
            on_click=clear_filters,
            help="Empties the profile and returns to the start.",
        )

    started = time.perf_counter()
    wanted = st.session_state.shown
    results = service.recommend(query, strategy=strategy, k=wanted)
    elapsed_ms = (time.perf_counter() - started) * 1000
    eligible = service.eligible_count(query, strategy)
    removed = service.excluded_counts(query, strategy)

    st.markdown(
        f'<div class="meta">Showing <b>{len(results)}</b> of '
        f"<b>{eligible:,}</b> matching products · {info.label} · "
        f"{elapsed_ms:.0f} ms end-to-end</div>",
        unsafe_allow_html=True,
    )
    if removed:
        total = sum(removed.values())
        with st.expander(f"Why {total:,} products were excluded from your results"):
            # One sentence per reason, each naming its own count. The previous
            # copy explained the filter-versus-ranking distinction in engineering
            # terms ("no score can bring them back — that is the point of a rule
            # rather than a preference") and stranded the number in a separate
            # line below it. Say what happened, then why it is removal rather
            # than demotion, in the reader's terms.
            for label, count in sorted(removed.items(), key=lambda kv: -kv[1]):
                st.markdown(f"We removed **{count:,} products** because {label}.")
            st.caption(
                "We don't just rank these lower — if a product isn't made for "
                "your skin, it shouldn't show up at all."
            )

    if not results:
        st.warning(
            "**Nothing matches.** Every candidate was ruled out by a filter rather "
            "than scored badly, so loosening one will help more than changing "
            "approach."
        )
        reasons = []
        if query.budget_max is not None:
            reasons.append(f"budget under ${query.budget_max:.0f}")
        if query.category:
            reasons.append(f"category *{query.category}*")
        if "sensitivity" in query.concerns:
            reasons.append("the sensitivity ingredient exclusions")
        if query.skin_type:
            reasons.append(
                f"products labelled for skin types other than {query.skin_type}"
            )
        if reasons:
            st.markdown("Active constraints: " + ", ".join(reasons) + ".")
        return

    if not personalised and profile["has_profile"]:
        st.info(
            f"You have entered profile details but this list is still the "
            f"unpersonalised one. Press **{FIND_LABEL}** to use them.",
            icon="👆",
        )
    elif personalised and query.is_cold_start:
        st.info(
            "Matched from your profile. Adding a product you already love is the "
            "single biggest improvement you can make to this list.",
            icon="💡",
        )

    st.write("")
    per_row = 5
    for start in range(0, len(results), per_row):
        row = results[start : start + per_row]
        for offset, (column, scored) in enumerate(
            zip(st.columns(per_row, gap="medium"), row)
        ):
            with column:
                render_card(service, scored, start + offset + 1)

    # Offered only when the last page came back full *and* the ceiling is not
    # reached. A short page means the eligible pool is exhausted, so there is
    # genuinely nothing more to show and a button would be dead.
    exhausted = len(results) < wanted
    st.write("")
    if not exhausted and len(results) < MAX_SHOWN:
        _, middle, _ = st.columns([2, 1, 2])
        with middle:
            st.button(f"Show {PAGE_SIZE} more", width="stretch", on_click=show_more)
    elif exhausted:
        st.caption(
            f"That is every one of the {len(results)} products matching your filters."
        )
    else:
        st.caption(
            f"Showing the top {MAX_SHOWN} of {eligible:,}. Rendering thousands of "
            "cards would make the page unusable, and rank 3,000 of a "
            "recommendation list tells you nothing — narrow the category, budget "
            "or concerns to get a shorter, sharper list."
        )

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    render_about(service, query)


if __name__ == "__main__":
    main()
