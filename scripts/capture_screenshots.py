from __future__ import annotations

from playwright.sync_api import Page, sync_playwright

from src.config import DELIVERABLES

BASE_URL = "http://localhost:8501"
OUTPUT = DELIVERABLES / "final UI screenshots"


def settle(page: Page, text: str) -> None:
    page.get_by_text(text, exact=False).first.wait_for(state="visible", timeout=30_000)
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(3_000)


def save(page: Page, name: str) -> None:
    page.screenshot(path=str(OUTPUT / name), full_page=False)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.goto(BASE_URL, wait_until="domcontentloaded", timeout=30_000)
        settle(page, "Student Success Dashboard")
        page.locator(".js-plotly-plot").first.wait_for(state="visible", timeout=30_000)
        save(page, "dashboard.png")

        page.get_by_test_id("stSidebarCollapseButton").locator("button").click()
        page.get_by_test_id("stExpandSidebarButton").wait_for(state="visible", timeout=10_000)
        page.wait_for_timeout(500)
        save(page, "dashboard_collapsed.png")
        page.get_by_test_id("stExpandSidebarButton").click()
        page.get_by_test_id("stSidebarNavItems").wait_for(state="visible", timeout=10_000)

        navigation = page.get_by_test_id("stSidebarNavItems")
        navigation.get_by_role("link", name="Student Assessment").click()
        settle(page, "Demonstration profiles")
        page.get_by_role("button", name="Strong Progress").wait_for(state="visible", timeout=30_000)
        page.get_by_role("button", name="Run Assessment").wait_for(state="visible", timeout=30_000)
        save(page, "student_assessment.png")
        page.get_by_role("button", name="Higher Support Need").click()
        page.wait_for_timeout(2_500)
        page.get_by_role("button", name="Run Assessment").click()
        settle(page, "Assessment ready for human review")
        page.locator(".risk-shell").scroll_into_view_if_needed()
        page.wait_for_timeout(500)
        save(page, "assessment_result.png")

        navigation.get_by_role("link", name="Cohort Analytics").click()
        settle(page, "Semester 1 average")
        save(page, "cohort_analytics.png")

        navigation.get_by_role("link", name="Model Performance").click()
        settle(page, "Five-model benchmark")
        save(page, "model_performance.png")
        browser.close()
    print(f"Captured final UI screenshots in {OUTPUT}")


if __name__ == "__main__":
    main()
