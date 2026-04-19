"""
Cross-platform Chrome/Selenium helpers.

On a local dev machine Chrome runs normally with a GUI. On a GCP/Linux VM we:
  - drop the `detach` experimental option (requires display)
  - add `--no-sandbox` and `--disable-dev-shm-usage` (Docker/Linux requirements)
  - point at the system `chromium` + `chromedriver` if FOBO_USE_SYSTEM_CHROME=true
  - wrap the call with Xvfb externally (via systemd or docker entrypoint) so the
    page sees a real display — Flashscore blocks pure --headless.

Set FOBO_CLOUD=true on the VM to activate cloud-safe defaults.
"""

import os
import shutil


def is_cloud():
    return os.environ.get("FOBO_CLOUD", "false").lower() == "true"


def apply_cloud_options(options):
    """
    Mutate a ChromeOptions object in place with Linux/Docker-safe flags.
    Safe to call on all platforms — on Windows/Mac these flags are harmless.
    """
    if not is_cloud():
        return options

    # Required in Docker / rootless Linux environments
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    # Don't set --headless — Flashscore blocks it. We rely on Xvfb.
    return options


def build_service():
    """
    Return a Selenium Service pointing at the right chromedriver.

    - Local: use webdriver-manager (downloads matching driver automatically)
    - Cloud with system chrome: use /usr/bin/chromedriver if present
    """
    from selenium.webdriver.chrome.service import Service

    if is_cloud():
        sys_driver = shutil.which("chromedriver") or "/usr/bin/chromedriver"
        if os.path.exists(sys_driver):
            return Service(sys_driver)

    # Fallback: webdriver-manager downloads correct version
    from webdriver_manager.chrome import ChromeDriverManager
    return Service(ChromeDriverManager().install())


def strip_incompatible_options(options):
    """
    Some experimental options (e.g. 'detach') require a real display.
    Strip them when running in cloud mode. Call after the scraper sets
    its preferred options but before constructing the driver.
    """
    if not is_cloud():
        return options

    # ChromeOptions stores experimental options in _experimental_options (private)
    # but we can safely rebuild by re-creating from scratch isn't trivial — instead
    # we mutate the internal dict if present.
    exp = getattr(options, "_experimental_options", None)
    if isinstance(exp, dict):
        exp.pop("detach", None)
    return options
