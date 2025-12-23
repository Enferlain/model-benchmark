import re
from playwright.sync_api import sync_playwright, expect

def run():
    """
    Performs an automated UI/UX check of the frontend's model URL validation and saves a verification screenshot.
    
    Opens a headless Chromium browser, navigates to http://localhost:3001, enters an invalid model URL into the URL input, clicks the "Download Model" button, and verifies that the validation message "Please enter a valid Civitai or HuggingFace model URL." appears. Prints success/failure messages to stdout, captures a screenshot to verification/ux_validation.png, and closes the browser. If the page fails to load, the function prints the navigation error and returns early.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        print("Navigating to Dashboard...")
        # Assuming frontend is on 3001 as seen in previous traces, backend on 8000
        # If frontend isn't running, we might need to start it, but let's assume it is based on memory.
        try:
            page.goto("http://localhost:3001", timeout=10000)
        except Exception as e:
            print(f"Failed to load page: {e}")
            return

        print("Page loaded. Testing error UX...")

        # 1. Find the URL input
        url_input = page.get_by_placeholder("https://...")
        expect(url_input).to_be_visible()

        # 2. Enter an invalid URL to trigger the validation error
        url_input.fill("invalid-url")

        # 3. Click Download
        download_btn = page.get_by_role("button", name="Download Model")
        download_btn.click()

        # 4. Expect the error message to appear (instead of alert)
        # The text we added is "Please enter a valid Civitai or HuggingFace model URL."
        error_msg = page.get_by_text("Please enter a valid Civitai or HuggingFace model URL.")
        try:
            expect(error_msg).to_be_visible(timeout=2000)
            print("SUCCESS: Error message is visible.")
        except AssertionError:
            print("FAILURE: Error message not found. Did the alert trigger instead?")

        # 5. Take screenshot
        page.screenshot(path="verification/ux_validation.png")
        print("Screenshot saved to verification/ux_validation.png")

        browser.close()

if __name__ == "__main__":
    run()