#!/bin/bash
# LinkedIn Automation CDP Wrapper Script
# Usage: ./cdp.sh <command> [args...]

# Ensure Node 22 is used
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22 > /dev/null 2>&1

CDP_SCRIPT="$HOME/.claude/skills/chrome-cdp/skills/chrome-cdp/scripts/cdp.mjs"

# Default LinkedIn tab target (update after running 'list')
LINKEDIN_TARGET="${LINKEDIN_TARGET:-}"

case "$1" in
  list)
    $CDP_SCRIPT list
    ;;

  nav|navigate)
    TARGET="${2:-$LINKEDIN_TARGET}"
    URL="${3:-https://www.linkedin.com/jobs/search/?keywords=Software%20Engineer&location=Taiwan&f_AL=true&f_TPR=r604800&f_WT=2,3}"
    $CDP_SCRIPT nav "$TARGET" "$URL"
    ;;

  shot|screenshot)
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT shot "$TARGET"
    echo "Screenshot saved to ~/.cache/cdp/"
    ;;

  snap|snapshot)
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT snap "$TARGET"
    ;;

  click)
    TARGET="${2:-$LINKEDIN_TARGET}"
    SELECTOR="$3"
    $CDP_SCRIPT click "$TARGET" "$SELECTOR"
    ;;

  clickxy)
    TARGET="${2:-$LINKEDIN_TARGET}"
    X="$3"
    Y="$4"
    $CDP_SCRIPT clickxy "$TARGET" "$X" "$Y"
    ;;

  type)
    TARGET="${2:-$LINKEDIN_TARGET}"
    TEXT="$3"
    $CDP_SCRIPT type "$TARGET" "$TEXT"
    ;;

  eval)
    TARGET="${2:-$LINKEDIN_TARGET}"
    JS="$3"
    $CDP_SCRIPT eval "$TARGET" "$JS"
    ;;

  apply)
    # Quick command to click Easy Apply button
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT click "$TARGET" "button.jobs-apply-button"
    ;;

  next)
    # Click Next button in modal
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT eval "$TARGET" '(() => {
      const modal = document.querySelector(".jobs-easy-apply-modal, [role=\"dialog\"]");
      if (!modal) return { error: "No modal" };
      const btns = modal.querySelectorAll("button");
      for (const btn of btns) {
        if (btn.textContent.toLowerCase().includes("next") && !btn.disabled) {
          btn.click();
          return { clicked: "Next" };
        }
      }
      return { error: "No Next button" };
    })()'
    ;;

  submit)
    # Click Submit button in modal
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT eval "$TARGET" '(() => {
      const modal = document.querySelector(".jobs-easy-apply-modal, [role=\"dialog\"]");
      if (!modal) return { error: "No modal" };
      const btns = modal.querySelectorAll("button");
      for (const btn of btns) {
        if (btn.textContent.toLowerCase().includes("submit") && !btn.disabled) {
          btn.click();
          return { clicked: "Submit" };
        }
      }
      return { error: "No Submit button" };
    })()'
    ;;

  close)
    # Close modal
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT click "$TARGET" "button[aria-label='Dismiss']"
    ;;

  review)
    # Click Review button in modal
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT click "$TARGET" "button[aria-label='Review your application']"
    ;;

  yes)
    # Answer "Yes" to work authorization questions
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT eval "$TARGET" '(() => {
      const modal = document.querySelector(".jobs-easy-apply-modal");
      if (!modal) return { error: "No modal" };
      const labels = modal.querySelectorAll("label");
      for (const label of labels) {
        if (label.textContent.trim() === "Yes") {
          label.click();
          return { clicked: "Yes" };
        }
      }
      return { error: "No Yes option" };
    })()'
    ;;

  nextjob)
    # Click on next unapplied job in list
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT eval "$TARGET" '(() => {
      const jobs = document.querySelectorAll("[data-job-id]");
      for (let i = 0; i < jobs.length; i++) {
        const job = jobs[i];
        if (!job.textContent.includes("Applied")) {
          const link = job.querySelector("a");
          if (link) {
            link.click();
            return { clicked: i, title: link.textContent.trim().substring(0, 50) };
          }
        }
      }
      return { error: "No unapplied jobs found" };
    })()'
    ;;

  auto)
    # Full auto-apply: next job -> easy apply -> all steps -> submit
    TARGET="${2:-$LINKEDIN_TARGET}"
    echo "Finding next unapplied job..."
    $0 nextjob "$TARGET"
    sleep 2
    echo "Clicking Easy Apply..."
    $0 apply "$TARGET"
    sleep 2
    echo "Processing application steps..."
    for i in 1 2 3 4 5; do
      $0 next "$TARGET" 2>/dev/null || true
      sleep 1.5
    done
    echo "Clicking Review..."
    $0 review "$TARGET" 2>/dev/null || true
    sleep 2
    echo "Submitting..."
    $0 submit "$TARGET"
    sleep 2
    echo "Closing modal..."
    $0 close "$TARGET"
    echo "Done!"
    ;;

  jobs)
    # List available jobs on page
    TARGET="${2:-$LINKEDIN_TARGET}"
    $CDP_SCRIPT eval "$TARGET" '(() => {
      const cards = [];
      document.querySelectorAll("[data-job-id]").forEach((el, i) => {
        const title = el.querySelector("a")?.textContent?.trim()?.substring(0, 60) || "Unknown";
        const company = el.querySelector(".job-card-container__company-name")?.textContent?.trim() || "";
        const applied = el.textContent.includes("Applied");
        cards.push({ index: i, title, company, applied });
      });
      return cards;
    })()'
    ;;

  help|*)
    echo "LinkedIn Automation CDP Wrapper"
    echo ""
    echo "Usage: ./cdp.sh <command> [target] [args...]"
    echo ""
    echo "Commands:"
    echo "  list              - List open Chrome tabs"
    echo "  nav <target> [url] - Navigate to URL (default: LinkedIn job search)"
    echo "  shot <target>     - Take screenshot"
    echo "  snap <target>     - Get accessibility snapshot"
    echo "  click <target> <selector> - Click element"
    echo "  clickxy <target> <x> <y>  - Click at coordinates"
    echo "  type <target> <text>      - Type text"
    echo "  eval <target> <js>        - Evaluate JavaScript"
    echo ""
    echo "Quick Commands:"
    echo "  jobs <target>     - List jobs on current page"
    echo "  nextjob <target>  - Click next unapplied job"
    echo "  apply <target>    - Click Easy Apply button"
    echo "  next <target>     - Click Next in modal"
    echo "  review <target>   - Click Review in modal"
    echo "  submit <target>   - Click Submit in modal"
    echo "  close <target>    - Close modal"
    echo "  yes <target>      - Answer Yes to work auth question"
    echo ""
    echo "Auto Commands:"
    echo "  auto <target>     - Full auto-apply (next job → apply → submit)"
    echo ""
    echo "Set LINKEDIN_TARGET env var to skip target arg:"
    echo "  export LINKEDIN_TARGET=4BF87A"
    ;;
esac
