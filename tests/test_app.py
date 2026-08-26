"""The Streamlit page itself, driven through Streamlit's own test harness.

Deliberately not screenshot tests. What matters is that the page renders the
things the assignment is graded on — a product grid, a reason per product, and an
honest empty state — and that changing an input actually changes the output. A UI
that silently ignores a control is the failure mode worth guarding, and it is
invisible to a screenshot.

Several tests pin *product* decisions rather than mechanics: that the landing
screen recommends nothing until asked, that model metrics stay out of the
shopping path, and that shopper-facing copy avoids jargon. Those are the things
most easily regressed while "improving" the page.

Widgets are looked up by label rather than index throughout. Index-based lookup
broke on every layout change, and the failure surfaced as a confusing type error
somewhere unrelated.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"

pytestmark = pytest.mark.skipif(
    not (ROOT / "artifacts" / "products.parquet").exists(),
    reason="artifacts not built; run scripts/03_build_artifacts.py",
)

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

TIMEOUT = 300

FIND = "✦  Find my products"
PAGE = 20          # app.PAGE_SIZE
CEILING = 100      # app.MAX_SHOWN
POPULAR = "Browse popular products instead"


@pytest.fixture(scope="module")
def owned_product_id() -> str:
    """A product id the history picker really offers, from the same source."""
    from src.service import RecommendationService

    service = RecommendationService.from_artifacts(ROOT / "artifacts")
    return service.history_pool(1)[0][0]


def run_app():
    app = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    app.run()
    return app


def widget(collection, label: str):
    for candidate in collection:
        if candidate.label == label:
            return candidate
    raise AssertionError(f"no widget {label!r}; have {[c.label for c in collection]}")


def button(app, label: str):
    return widget(app.button, label)


def product_ids(app) -> list[str]:
    found = [re.search(r"`(P\d+)`", c.value) for c in app.caption]
    return [m.group(1) for m in found if m]


def card_names(app) -> list[str]:
    return [
        m.group(1)
        for m in (
            re.search(r'class="pname">([^<]+)', block.value) for block in app.markdown
        )
        if m
    ]


def chips(app) -> list[str]:
    """The profile summary chips shown above the results."""
    return re.findall(r'class="chip[^"]*">([^<]+)', all_text(app))


def meta_line(app) -> str:
    """The "10 recommendations - Best overall - 22 ms" strip, tags stripped."""
    for block in app.markdown:
        if 'class="meta"' in block.value:
            return re.sub(r"<[^>]+>", "", block.value)
    return ""


def is_personalised(app) -> bool:
    return bool(chips(app)) and "Not personalised" not in chips(app)


def all_text(app) -> str:
    return " ".join(m.value for m in app.markdown)


def show_results(app, skin_type: str = "dry"):
    """Enter a profile and ask for recommendations."""
    widget(app.sidebar.radio, "Skin type").set_value(skin_type)
    app.run()
    button(app, FIND).click()
    app.run()
    return app


# --------------------------------------------------------------------------
# The landing screen recommends nothing
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def landing():
    return run_app()


def test_the_page_loads_without_raising(landing):
    assert not landing.exception, (
        landing.exception[0].message if landing.exception else ""
    )


def test_the_landing_screen_shows_no_products(landing):
    """The point of the landing screen.

    Showing ten products before the visitor has said anything, then explaining
    underneath that they are not personalised, spends the most valuable screen on
    a list identical for everybody.
    """
    assert product_ids(landing) == []
    assert card_names(landing) == []


def test_the_profile_panel_lives_in_the_sidebar(landing):
    """It stays put while results scroll, which is why a filter rail exists."""
    assert widget(landing.sidebar.radio, "Skin type")
    assert widget(landing.sidebar.pills, "Concerns")
    assert widget(landing.sidebar.selectbox, "Skin tone")
    assert widget(landing.sidebar.multiselect, "Products you already love")
    assert widget(landing.sidebar.select_slider, "Budget")


def test_the_sidebar_is_grouped_into_labelled_sections(landing):
    text = " ".join(m.value for m in landing.sidebar.markdown)
    for section in ("Your skin", "What you want to improve", "Narrow it down",
                    "Your preferences"):
        assert section in text, f"missing sidebar section {section!r}"


def test_the_landing_screen_invites_a_profile(landing):
    assert "Tell us about your skin" in all_text(landing)
    assert "fit <em>your</em> skin" in all_text(landing)


def test_the_landing_screen_explains_the_flow(landing):
    """01 / 02 / 03 onboarding rather than a wall of products."""
    assert all(step in all_text(landing) for step in ("01", "02", "03"))
    assert len(re.findall(r'class="stepcol"', all_text(landing))) == 3


def test_the_primary_action_is_disabled_until_something_is_entered(landing):
    assert button(landing, FIND).disabled
    assert any(
        "Pick a skin type" in c.value for c in landing.sidebar.caption
    ), "the disabled button needs to say why"


def test_the_landing_screen_offers_a_way_past_personalisation(landing):
    assert button(landing, POPULAR)


def test_the_landing_screen_does_not_claim_a_recommendation(landing):
    assert chips(landing) == []
    assert meta_line(landing) == ""


def test_the_catalogue_scale_is_stated_up_front(landing):
    assert "Products" in all_text(landing) and "Real ratings" in all_text(landing)


# --------------------------------------------------------------------------
# Entering a profile
# --------------------------------------------------------------------------


def test_entering_a_skin_type_enables_the_primary_action():
    app = run_app()
    widget(app.sidebar.radio, "Skin type").set_value("oily")
    app.run()
    assert not button(app, FIND).disabled


def test_asking_for_recommendations_shows_a_personalised_grid():
    app = show_results(run_app())
    assert not app.exception
    assert len(product_ids(app)) == PAGE
    assert is_personalised(app)
    assert "Dry skin" in chips(app)


def test_the_results_view_summarises_the_profile_as_chips():
    app = show_results(run_app())
    assert "Dry skin" in chips(app)
    assert any("Under $" in c for c in chips(app))


def test_the_results_view_reports_count_model_and_latency():
    """The strip an evaluator reads to confirm the thing actually ran."""
    app = show_results(run_app())
    line = meta_line(app)
    assert "Showing" in line and str(PAGE) in line
    assert "Best overall" in line
    assert re.search(r"\d+ ms", line), line
    assert "matching products" in line


def test_exploring_popular_products_is_labelled_as_not_personalised():
    app = run_app()
    button(app, POPULAR).click()
    app.run()
    assert len(product_ids(app)) == PAGE
    assert not is_personalised(app)
    assert "Not personalised" in chips(app)


def test_a_concern_alone_is_enough_to_personalise():
    app = run_app()
    widget(app.sidebar.pills, "Concerns").set_value(["acne"])
    app.run()
    button(app, FIND).click()
    app.run()
    assert is_personalised(app)


# --------------------------------------------------------------------------
# Reasoning, which the assignment grades explicitly
# --------------------------------------------------------------------------


def test_each_product_carries_a_visible_reason():
    app = show_results(run_app())
    evidence = [m for m in app.markdown if 'class="ev' in m.value]
    assert len(evidence) >= PAGE
    assert [e.label for e in app.expander].count("Why this?") == PAGE


def test_unmet_criteria_are_shown_not_hidden():
    """A checklist of nothing but ticks is marketing."""
    app = run_app()
    widget(app.sidebar.radio, "Skin type").set_value("dry")
    widget(app.sidebar.pills, "Concerns").set_value(
        ["dryness", "dark_circles"]
    )
    app.run()
    button(app, FIND).click()
    app.run()

    blocks = [m.value for m in app.markdown if 'class="ev' in m.value]
    assert any('class="ev yes"' in b for b in blocks), "no met criteria shown"
    assert any('class="ev no"' in b for b in blocks), "no unmet criteria shown"


def test_score_components_use_shopper_language():
    app = show_results(run_app())
    text = all_text(app)
    assert "Similar shoppers" in text
    assert "Product match" in text


# --------------------------------------------------------------------------
# Product decisions worth pinning
# --------------------------------------------------------------------------


def test_model_metrics_stay_out_of_the_shopping_path():
    """An evaluator should find NDCG easily; a shopper should never meet it."""
    app = show_results(run_app())
    assert "NDCG" not in meta_line(app)
    assert not any("NDCG" in chip for chip in chips(app))


def test_the_shopping_column_avoids_jargon():
    """Nobody should have to learn the word "cohort" to use this."""
    app = show_results(run_app())
    jargon = ("cohort", "TF-IDF", "MMR", "item-item", "NDCG")
    for surface in [meta_line(app), *chips(app)]:
        for term in jargon:
            assert term not in surface, f"{term!r} leaked into shopper copy"


def test_the_technical_detail_is_still_available_somewhere():
    """Hiding jargon must not mean deleting it — evaluators need it."""
    app = show_results(run_app())
    assert any("How this works" in e.label for e in app.expander)


def test_the_approach_selector_stays_available_for_comparison():
    app = run_app()
    assert widget(app.sidebar.selectbox, "Approach")


def test_the_page_never_renders_a_stray_module_docstring():
    """Streamlit's magic prints any bare top-level expression as page content.

    A documentation string written loose at module level therefore leaks onto the
    page as body text, which is how a note about `review_count` once ended up
    above the title.
    """
    import ast

    tree = ast.parse(APP.read_text())
    stray = [
        node.lineno
        for node in tree.body[1:]  # body[0] is the real module docstring
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert not stray, f"bare strings at app.py lines {stray} will render on the page"


# --------------------------------------------------------------------------
# Controls must actually do something
# --------------------------------------------------------------------------


def test_declaring_products_you_love_changes_the_list(owned_product_id):
    app = show_results(run_app())
    before = product_ids(app)

    # AppTest wants the raw option value here; `.options` hands back the
    # format_func output, so it cannot be round-tripped into set_value.
    widget(app.sidebar.multiselect, "Products you already love").set_value([owned_product_id])
    app.run()
    assert not app.exception
    assert widget(app.sidebar.multiselect, "Products you already love").value == [
        owned_product_id
    ], "the widget ignored the value"
    assert product_ids(app) != before


def test_switching_approach_changes_the_ranking():
    app = show_results(run_app())
    hybrid = product_ids(app)
    widget(app.sidebar.selectbox, "Approach").set_value("popularity")
    app.run()
    assert product_ids(app) != hybrid


def eligible_count(app) -> int:
    """The "Showing 10 of N matching products" figure from the meta strip."""
    match = re.search(r"of ([\d,]+) matching products", meta_line(app))
    assert match, f"no eligible count in {meta_line(app)!r}"
    return int(match.group(1).replace(",", ""))


def test_concerns_narrow_the_pool():
    app = show_results(run_app())
    wide = eligible_count(app)

    widget(app.sidebar.pills, "Concerns").set_value(["sensitivity"])
    app.run()
    assert eligible_count(app) < wide


def test_the_result_count_grows_past_one_page():
    """The old "number of results" slider stopped dead at 20."""
    app = show_results(run_app())
    assert len(product_ids(app)) == PAGE

    button(app, f"Show {PAGE} more").click()
    app.run()
    assert len(product_ids(app)) == 2 * PAGE, "paging stopped at the old ceiling"


def test_paging_stops_at_the_documented_ceiling_and_explains_itself():
    """A ceiling exists — ~45,000 elements would hang the page — so it must say
    so and point somewhere useful rather than silently offering a dead button."""
    app = show_results(run_app())
    for _ in range(CEILING // PAGE):
        remaining = [b for b in app.button if b.label == f"Show {PAGE} more"]
        if not remaining:
            break
        remaining[0].click()
        app.run()

    assert len(product_ids(app)) == CEILING
    assert f"Show {PAGE} more" not in [b.label for b in app.button]
    assert any(f"top {CEILING}" in c.value for c in app.caption), (
        "the ceiling must explain itself"
    )
    assert any("narrow" in c.value for c in app.caption)


def test_paging_stops_when_the_eligible_pool_runs_out():
    """A short page means there is genuinely nothing more, so no dead button."""
    app = run_app()
    widget(app.sidebar.radio, "Skin type").set_value("dry")
    widget(app.sidebar.selectbox, "Category").set_value("Beauty Supplements")
    app.run()
    button(app, FIND).click()
    app.run()

    shown = len(product_ids(app))
    assert 0 < shown < PAGE, f"expected a small eligible pool, got {shown}"
    assert f"Show {PAGE} more" not in [b.label for b in app.button]
    assert any("every one of" in c.value for c in app.caption)


def test_a_new_profile_resets_paging():
    app = show_results(run_app())
    button(app, f"Show {PAGE} more").click()
    app.run()
    assert len(product_ids(app)) == 2 * PAGE

    widget(app.sidebar.radio, "Skin type").set_value("oily")
    app.run()
    button(app, FIND).click()
    app.run()
    assert len(product_ids(app)) == PAGE, "paging should restart for a new profile"


def test_impossible_constraints_produce_an_honest_empty_state():
    app = show_results(run_app())
    widget(app.sidebar.selectbox, "Category").set_value("Beauty Supplements")
    widget(app.sidebar.select_slider, "Budget").set_value(25)
    app.run()

    assert not app.exception
    assert product_ids(app) == []
    assert app.warning and "Nothing matches" in app.warning[0].value
    constraints = [m.value for m in app.markdown if "Active constraints" in m.value]
    assert constraints and "Beauty Supplements" in constraints[0]


def test_popular_mode_does_not_secretly_use_your_profile():
    """A list labelled "the same for everybody" must actually be that.

    The profile widgets stay reachable in popular mode, so it would be easy to
    keep feeding them into the query and quietly personalise a list the page has
    just described as unpersonalised.
    """
    app = run_app()
    button(app, POPULAR).click()
    app.run()
    baseline = product_ids(app)

    widget(app.sidebar.radio, "Skin type").set_value("oily")
    widget(app.sidebar.pills, "Concerns").set_value(["acne"])
    app.run()

    assert not is_personalised(app)
    assert product_ids(app) == baseline, "the profile leaked into an unpersonalised list"
    assert app.info and "Press" in app.info[0].value, "should nudge toward the button"


def test_pressing_the_button_then_applies_that_same_profile():
    app = run_app()
    button(app, POPULAR).click()
    app.run()
    popular = product_ids(app)

    widget(app.sidebar.radio, "Skin type").set_value("oily")
    widget(app.sidebar.pills, "Concerns").set_value(["acne"])
    app.run()
    button(app, FIND).click()
    app.run()

    assert is_personalised(app)
    assert product_ids(app) != popular


# --------------------------------------------------------------------------
# Regressions from a round of visual bugs
# --------------------------------------------------------------------------


def test_css_does_not_blanket_override_font_family():
    """The cause of "arrow_right" printing on top of expander labels.

    Streamlit renders expander chevrons as Material icon spans whose text
    content is the ligature name. Forcing font-family onto every div and span
    hits those too, and the icon font never gets a chance to turn the name into
    a glyph — so the label and the word "arrow_right" overlap.
    """
    css = APP.read_text()
    css = css[css.index("CSS = ") : css.index("</style>")]
    offenders = [
        line.strip()
        for line in css.splitlines()
        if "font-family" in line and "Material" not in line and "@import" not in line
    ]
    # Scoped class selectors are fine; generic element selectors are not.
    for line in offenders:
        assert ".stApp div" not in line and ".stApp span" not in line, line


def test_both_palettes_are_specified_in_full():
    """Light and dark each declare their own values rather than half-inheriting."""
    config = (ROOT / ".streamlit" / "config.toml").read_text()

    def section(name: str) -> str:
        match = re.search(rf"^\[{re.escape(name)}\]$(.*?)(?=^\[|\Z)", config,
                          re.M | re.S)
        assert match, f"no [{name}] section"
        return match.group(1)

    for mode in ("light", "dark"):
        block = section(f"theme.{mode}")
        for option in ("primaryColor", "backgroundColor", "textColor", "borderColor"):
            assert option in block, f"[theme.{mode}] is missing {option}"
        assert "backgroundColor" in section(f"theme.{mode}.sidebar")


def test_the_reader_can_still_choose_a_theme():
    """"minimal" would remove the light/dark control along with the menu."""
    config = (ROOT / ".streamlit" / "config.toml").read_text()
    mode = re.search(r'toolbarMode\s*=\s*"(\w+)"', config)
    assert mode and mode.group(1) != "minimal", "the theme control is gone"


def test_custom_css_follows_streamlit_not_the_operating_system():
    """The desync that produced grey text on white.

    `prefers-color-scheme` reports the OS, so custom CSS keyed to it disagreed
    with an explicit in-app Light choice: Streamlit repainted white while the
    custom markup stayed dark. Tokens must key off the measured appearance
    instead.
    """
    css = APP.read_text()
    css = css[css.index("CSS = ") : css.index("</style>")]
    assert "@media (prefers-color-scheme" not in css, "keyed to the OS again"
    assert ':root[data-appearance="dark"]' in css
    assert ':root[data-appearance="light"]' in css


def test_every_token_has_a_value_in_both_appearances():
    css = APP.read_text()
    light = css[css.index(':root, :root[data-appearance="light"]') :]
    light = light[: light.index("}")]
    dark = css[css.index(':root[data-appearance="dark"] {') :]
    dark = dark[: dark.index("}")]
    tokens = set(re.findall(r"(--[a-z-]+):", light))
    assert tokens, "no tokens found"
    missing = {t for t in tokens if t not in dark}
    assert not missing, f"tokens with no dark value: {sorted(missing)}"


def test_print_and_record_screen_are_hidden_without_losing_the_toggle():
    """What was actually asked for, twice.

    Streamlit's `toolbarMode` can only drop the entire viewer menu, which takes
    the light/dark control with it. These two entries carry a stable testid, so
    the injected script hides exactly them and leaves the rest of the menu alone.
    """
    source = APP.read_text()
    assert "THEME_SYNC" in source
    assert 'st.markdown(CSS, unsafe_allow_html=True)\n    components.html(THEME_SYNC' in source, (
        "the script is defined but never rendered"
    )
    for entry in ('"Print"', '"Record screen"'):
        assert entry in source, f"{entry} is not being hidden"
    assert 'data-testid="stMainMenuItemLabel"' in source
    # The menu popover is built on demand, so a one-shot pass would miss it.
    assert "MutationObserver" in source


def test_the_injected_script_survives_python_string_escaping():
    """Two layers of Python quoting sit between the source and the browser."""
    source = APP.read_text()
    namespace: dict = {}
    exec(
        compile(
            source[source.index("THEME_SYNC = ") : source.index("CSS = ")],
            "theme_sync",
            "exec",
        ),
        namespace,
    )
    script = namespace["THEME_SYNC"]
    assert "/[\\d.]+/g" in script, "the number class arrived mangled"
    assert "window.parent" in script
    assert "0.299" in script, "luminance maths missing"
    # The bug this guards: a transparent element computes to rgba(0,0,0,0), and
    # reading those first three numbers as a colour says "black".
    assert "alpha > 0.05" in script, "transparent backgrounds unguarded"
    assert "el.parentElement" in script, "does not walk up for a painted ancestor"


def test_clear_filters_empties_the_profile_and_returns_to_onboarding():
    app = show_results(run_app())
    assert product_ids(app)

    button(app, "Clear filters").click()
    app.run()

    assert not app.exception
    assert product_ids(app) == []
    assert len(re.findall(r'class="stepcol"', all_text(app))) == 3
    assert widget(app.sidebar.radio, "Skin type").value is None
    assert widget(app.sidebar.pills, "Concerns").value == []


def test_clear_filters_sits_beside_the_results_not_at_the_bottom():
    app = show_results(run_app())
    labels = [b.label for b in app.button]
    assert "Clear filters" in labels
    assert "Start over" not in labels, "the stranded bottom button should be gone"


def test_each_card_face_renders_as_one_block():
    """Per-element rendering made otherwise identical cards different heights,
    because Streamlit inserts a gap between elements."""
    app = show_results(run_app())
    faces = re.findall(r'class="pface"', all_text(app))
    plates = re.findall(r'class="plate t-\w+"', all_text(app))
    boxes = re.findall(r'class="evbox"', all_text(app))
    assert len(faces) == len(plates) == len(boxes) == PAGE


def test_plate_tints_are_category_derived_not_random():
    app = show_results(run_app())
    classes = set(re.findall(r'class="plate (t-\w+)"', all_text(app)))
    assert classes, "no tint classes rendered"
    assert all(c.startswith("t-") for c in classes)
    # A hash-per-product scheme would produce ~10 distinct classes for 10 cards.
    assert len(classes) <= 9, f"tints look per-product, not per-category: {classes}"


# --------------------------------------------------------------------------
# Popular mode, and the crash it exposed
# --------------------------------------------------------------------------


def test_a_product_with_no_reviews_does_not_crash_the_page():
    """278 products (3.3%) carry no review count.

    `int(x or 0)` does not defend against that: NaN is truthy, so it passes
    straight through and int() raises. Only strategies that can surface an
    unreviewed product hit it — random and content-only — which is why the
    popularity-weighted default never showed it.
    """
    app = show_results(run_app())
    for approach in ("random", "content"):
        widget(app.sidebar.selectbox, "Approach").set_value(approach)
        app.run()
        assert not app.exception, (
            f"{approach} crashed: "
            f"{app.exception[0].message if app.exception else ''}"
        )
        assert product_ids(app), f"{approach} returned nothing"


def test_popular_mode_is_actually_ranked_by_popularity():
    """A heading that says "Popular right now" has to mean it.

    Popular mode used to read the sidebar's Approach, so choosing Random and
    pressing Browse popular produced random products under that heading.
    """
    from src.schema import Query
    from src.service import RecommendationService

    service = RecommendationService.from_artifacts(ROOT / "artifacts")
    expected = [
        r.product_id
        for r in service.recommend(Query(budget_max=100.0), "popularity", PAGE)
    ]

    for approach in (None, "random", "content", "cohort-cf"):
        app = run_app()
        if approach:
            widget(app.sidebar.selectbox, "Approach").set_value(approach)
            app.run()
        button(app, POPULAR).click()
        app.run()

        assert not app.exception
        assert product_ids(app) == expected, (
            f"Approach={approach} changed a list labelled popular"
        )
        assert "Most popular" in meta_line(app)


def test_the_approach_selector_still_applies_to_personalised_results():
    """The override must be scoped to popular mode, not global."""
    app = show_results(run_app())
    baseline = product_ids(app)
    widget(app.sidebar.selectbox, "Approach").set_value("popularity")
    app.run()
    assert product_ids(app) != baseline
    assert "Most popular" in meta_line(app)
