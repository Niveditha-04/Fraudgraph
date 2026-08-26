"""
End-to-end consistency check for the live dashboard.

Runs against the actual deployed URL, not local static files -- a local-file
test would not catch the class of bug this is regression coverage for: a
CDN in front of the deployed site serving different cached ages of
cases.json vs case_N.json to different sessions, with no Cache-Control
header to make that behavior visible from the response alone. Only a test
that loads the real deployed page and clicks through it exercises that.

For each of the 10 cases, clicks the sidebar entry and asserts three
independently-sourced pieces of data agree: the sidebar list's own
node id and hybrid score, the detail panel's score tile, and the wallet
number the memo text itself quotes. All three are rendered from separate
fetches (cases.json, case_N.json, and the memo field within case_N.json)
so an agreement across all three is a real consistency check, not a
tautology.
"""
import re

import pytest
from playwright.sync_api import expect, sync_playwright

DASHBOARD_URL = "https://niv04-fraudgraph.static.hf.space/"
N_CASES = 10


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        pg = browser.new_page()
        pg.goto(DASHBOARD_URL, wait_until="networkidle")
        yield pg
        browser.close()


@pytest.mark.parametrize("case_id", range(N_CASES))
def test_case_data_consistent_across_sidebar_detail_and_memo(page, case_id):
    sidebar_item = page.locator(f"#case-item-{case_id}")

    sidebar_text = sidebar_item.inner_text()
    node_match = re.search(r"node (\d+)", sidebar_text)
    hybrid_match = re.search(r"hybrid ([\d.]+)", sidebar_text)
    assert node_match, f"case {case_id}: sidebar entry has no node id -- {sidebar_text!r}"
    assert hybrid_match, f"case {case_id}: sidebar entry has no hybrid score -- {sidebar_text!r}"
    sidebar_node = node_match.group(1)
    sidebar_hybrid = float(hybrid_match.group(1))

    sidebar_item.click()

    detail = page.locator("#detail")
    memo_locator = detail.locator(".memo")
    # The memo div from the PREVIOUS case is already visible before this
    # click -- to_be_visible() alone would pass instantly against stale
    # content. Wait for the memo to actually contain this case's own node
    # id, which only becomes true once the new case_N.json fetch resolves
    # and re-renders.
    expect(memo_locator).to_contain_text(sidebar_node, timeout=10_000)

    score_tiles = detail.locator(".score-tile .value")
    detail_hybrid = float(score_tiles.nth(0).inner_text())

    memo_text = memo_locator.inner_text()
    memo_match = re.search(r"[Ww]allet(?: [Cc]ase)? ?#(\d+)", memo_text)
    assert memo_match, f"case {case_id}: memo text has no quoted wallet number -- {memo_text[:200]!r}"
    memo_node = memo_match.group(1)

    assert sidebar_node == memo_node, (
        f"case {case_id}: sidebar shows node {sidebar_node} but memo quotes wallet #{memo_node} -- "
        "sidebar and detail panel are out of sync"
    )
    assert abs(sidebar_hybrid - detail_hybrid) < 0.01, (
        f"case {case_id}: sidebar hybrid score {sidebar_hybrid} does not match detail panel's "
        f"{detail_hybrid} -- sidebar and detail panel are out of sync"
    )
