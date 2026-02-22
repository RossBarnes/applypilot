"""ApplyPilot CLI — the main entry point."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from applypilot import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

app = typer.Typer(
    name="applypilot",
    help="AI-powered end-to-end job application pipeline.",
    no_args_is_help=True,
)
console = Console()
log = logging.getLogger(__name__)

# Valid pipeline stages (in execution order)
VALID_STAGES = ("discover", "enrich", "score", "tailor", "cover", "pdf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap() -> None:
    """Common setup: load env, create dirs, init DB."""
    from applypilot.config import load_env, ensure_dirs
    from applypilot.database import init_db

    load_env()
    ensure_dirs()
    init_db()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold]applypilot[/bold] {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """ApplyPilot — AI-powered end-to-end job application pipeline."""


@app.command()
def init() -> None:
    """Run the first-time setup wizard (profile, resume, search config)."""
    from applypilot.wizard.init import run_wizard

    run_wizard()


@app.command()
def run(
    stages: Optional[list[str]] = typer.Argument(
        None,
        help=(
            "Pipeline stages to run. "
            f"Valid: {', '.join(VALID_STAGES)}, all. "
            "Defaults to 'all' if omitted."
        ),
    ),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for tailor/cover stages."),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel threads for discovery/enrichment stages."),
    stream: bool = typer.Option(False, "--stream", help="Run stages concurrently (streaming mode)."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview stages without executing."),
) -> None:
    """Run pipeline stages. Defaults to discover, enrich, score. Use 'all' for the full pipeline including tailoring and auto-apply prep."""
    _bootstrap()

    from applypilot.pipeline import run_pipeline

    stage_list = stages if stages else ["discover", "enrich", "score"]

    # Validate stage names
    for s in stage_list:
        if s != "all" and s not in VALID_STAGES:
            console.print(
                f"[red]Unknown stage:[/red] '{s}'. "
                f"Valid stages: {', '.join(VALID_STAGES)}, all"
            )
            raise typer.Exit(code=1)

    # Gate AI stages behind Tier 2
    llm_stages = {"score", "tailor", "cover"}
    if any(s in stage_list for s in llm_stages) or "all" in stage_list:
        from applypilot.config import check_tier
        check_tier(2, "AI scoring/tailoring")

    result = run_pipeline(
        stages=stage_list,
        min_score=min_score,
        dry_run=dry_run,
        stream=stream,
        workers=workers,
    )

    if result.get("errors"):
        raise typer.Exit(code=1)


@app.command()
def apply(
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Max applications to submit."),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel browser workers."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score for job selection."),
    model: str = typer.Option("haiku", "--model", "-m", help="Claude model name."),
    continuous: bool = typer.Option(False, "--continuous", "-c", help="Run forever, polling for new jobs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview actions without submitting."),
    headless: bool = typer.Option(False, "--headless", help="Run browsers in headless mode."),
    url: Optional[str] = typer.Option(None, "--url", help="Apply to a specific job URL."),
    gen: bool = typer.Option(False, "--gen", help="Generate prompt file for manual debugging instead of running."),
    mark_applied: Optional[str] = typer.Option(None, "--mark-applied", help="Manually mark a job URL as applied."),
    mark_failed: Optional[str] = typer.Option(None, "--mark-failed", help="Manually mark a job URL as failed (provide URL)."),
    fail_reason: Optional[str] = typer.Option(None, "--fail-reason", help="Reason for --mark-failed."),
    reset_failed: bool = typer.Option(False, "--reset-failed", help="Reset all failed jobs for retry."),
) -> None:
    """Launch auto-apply to submit job applications."""
    _bootstrap()

    from applypilot.config import check_tier, PROFILE_PATH as _profile_path
    from applypilot.database import get_connection

    # --- Utility modes (no Chrome/Claude needed) ---

    if mark_applied:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_applied, "applied")
        console.print(f"[green]Marked as applied:[/green] {mark_applied}")
        return

    if mark_failed:
        from applypilot.apply.launcher import mark_job
        mark_job(mark_failed, "failed", reason=fail_reason)
        console.print(f"[yellow]Marked as failed:[/yellow] {mark_failed} ({fail_reason or 'manual'})")
        return

    if reset_failed:
        from applypilot.apply.launcher import reset_failed as do_reset
        count = do_reset()
        console.print(f"[green]Reset {count} failed job(s) for retry.[/green]")
        return

    # --- Full apply mode ---

    # Check 1: Tier 3 required (Claude Code CLI + Chrome)
    check_tier(3, "auto-apply")

    # Check 2: Profile exists
    if not _profile_path.exists():
        console.print(
            "[red]Profile not found.[/red]\n"
            "Run [bold]applypilot init[/bold] to create your profile first."
        )
        raise typer.Exit(code=1)

    # Check 3: Tailored resumes exist (skip for --gen with --url)
    if not (gen and url):
        conn = get_connection()
        ready = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE tailored_resume_path IS NOT NULL AND applied_at IS NULL"
        ).fetchone()[0]
        if ready == 0:
            console.print(
                "[red]No tailored resumes ready.[/red]\n"
                "Run [bold]applypilot run score tailor[/bold] first to prepare applications."
            )
            raise typer.Exit(code=1)

    if gen:
        from applypilot.apply.launcher import gen_prompt, BASE_CDP_PORT
        target = url or ""
        if not target:
            console.print("[red]--gen requires --url to specify which job.[/red]")
            raise typer.Exit(code=1)
        prompt_file = gen_prompt(target, min_score=min_score, model=model)
        if not prompt_file:
            console.print("[red]No matching job found for that URL.[/red]")
            raise typer.Exit(code=1)
        mcp_path = _profile_path.parent / ".mcp-apply-0.json"
        console.print(f"[green]Wrote prompt to:[/green] {prompt_file}")
        console.print(f"\n[bold]Run manually:[/bold]")
        console.print(
            f"  claude --model {model} -p "
            f"--mcp-config {mcp_path} "
            f"--permission-mode bypassPermissions < {prompt_file}"
        )
        return

    from applypilot.apply.launcher import main as apply_main

    effective_limit = limit if limit is not None else (0 if continuous else 1)

    console.print("\n[bold blue]Launching Auto-Apply[/bold blue]")
    console.print(f"  Limit:    {'unlimited' if continuous else effective_limit}")
    console.print(f"  Workers:  {workers}")
    console.print(f"  Model:    {model}")
    console.print(f"  Headless: {headless}")
    console.print(f"  Dry run:  {dry_run}")
    if url:
        console.print(f"  Target:   {url}")
    console.print()

    apply_main(
        limit=effective_limit,
        target_url=url,
        min_score=min_score,
        headless=headless,
        model=model,
        dry_run=dry_run,
        continuous=continuous,
        workers=workers,
    )


@app.command()
def status() -> None:
    """Show pipeline statistics from the database."""
    _bootstrap()

    from applypilot.database import get_stats

    stats = get_stats()

    console.print("\n[bold]ApplyPilot Pipeline Status[/bold]\n")

    # Summary table
    summary = Table(title="Pipeline Overview", show_header=True, header_style="bold cyan")
    summary.add_column("Metric", style="bold")
    summary.add_column("Count", justify="right")

    summary.add_row("Total jobs discovered", str(stats["total"]))
    summary.add_row("With full description", str(stats["with_description"]))
    summary.add_row("Pending enrichment", str(stats["pending_detail"]))
    summary.add_row("Enrichment errors", str(stats["detail_errors"]))
    summary.add_row("Scored by LLM", str(stats["scored"]))
    summary.add_row("Pending scoring", str(stats["unscored"]))
    summary.add_row("Tailored resumes", str(stats["tailored"]))
    summary.add_row("Pending tailoring (7+)", str(stats["untailored_eligible"]))
    summary.add_row("Cover letters", str(stats["with_cover_letter"]))
    summary.add_row("Ready to apply", str(stats["ready_to_apply"]))
    summary.add_row("Applied", str(stats["applied"]))
    summary.add_row("Apply errors", str(stats["apply_errors"]))

    console.print(summary)

    # Score distribution
    if stats["score_distribution"]:
        dist_table = Table(title="\nScore Distribution", show_header=True, header_style="bold yellow")
        dist_table.add_column("Score", justify="center")
        dist_table.add_column("Count", justify="right")
        dist_table.add_column("Bar")

        max_count = max(count for _, count in stats["score_distribution"]) or 1
        for score, count in stats["score_distribution"]:
            bar_len = int(count / max_count * 30)
            if score >= 7:
                color = "green"
            elif score >= 5:
                color = "yellow"
            else:
                color = "red"
            bar = f"[{color}]{'=' * bar_len}[/{color}]"
            dist_table.add_row(str(score), str(count), bar)

        console.print(dist_table)

    # By site
    if stats["by_site"]:
        site_table = Table(title="\nJobs by Source", show_header=True, header_style="bold magenta")
        site_table.add_column("Site")
        site_table.add_column("Count", justify="right")

        for site, count in stats["by_site"]:
            site_table.add_row(site or "Unknown", str(count))

        console.print(site_table)

    console.print()


@app.command()
def dashboard() -> None:
    """Generate and open the HTML dashboard in your browser."""
    _bootstrap()

    from applypilot.view import open_dashboard

    open_dashboard()


@app.command()
def schedule(
    remove: bool = typer.Option(False, "--remove", help="Remove the scheduled pipeline run."),
    day: str = typer.Option("sunday", "--day", help="Day of week to run (e.g. sunday, monday)."),
    time: str = typer.Option("20:00", "--time", help="Time to run in HH:MM 24h format."),
) -> None:
    """Install (or remove) a cron job to run the full pipeline automatically."""
    import shutil
    import subprocess
    import sys

    if not shutil.which("crontab"):
        console.print(
            "[red]crontab not available.[/red] "
            "On Windows, use Task Scheduler instead:\n"
            "  schtasks /create /tn ApplyPilot /tr \"applypilot run all\" /sc WEEKLY /d SUN /st 20:00"
        )
        raise typer.Exit(code=1)

    MARKER = "# applypilot-schedule"

    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    existing = result.stdout if result.returncode == 0 else ""
    lines = [l for l in existing.splitlines() if MARKER not in l]

    if remove:
        new_crontab = "\n".join(lines) + ("\n" if lines else "")
        subprocess.run(["crontab", "-"], input=new_crontab, text=True, check=True)
        console.print("[green]Schedule removed.[/green]")
        return

    day_map = {
        "sunday": 0, "sun": 0,
        "monday": 1, "mon": 1,
        "tuesday": 2, "tue": 2,
        "wednesday": 3, "wed": 3,
        "thursday": 4, "thu": 4,
        "friday": 5, "fri": 5,
        "saturday": 6, "sat": 6,
    }
    day_num = day_map.get(day.lower())
    if day_num is None:
        console.print(f"[red]Unknown day:[/red] '{day}'. Use e.g. sunday, monday.")
        raise typer.Exit(code=1)

    try:
        hour, minute = time.split(":")
        hour, minute = int(hour), int(minute)
    except ValueError:
        console.print(f"[red]Invalid time:[/red] '{time}'. Use HH:MM (e.g. 20:00).")
        raise typer.Exit(code=1)

    from applypilot.config import APP_DIR, LOG_DIR
    binary = shutil.which("applypilot") or f"{sys.executable} -m applypilot"
    log_file = LOG_DIR / "cron.log"

    cron_line = f"{minute} {hour} * * {day_num} {binary} run all >> {log_file} 2>&1  {MARKER}"
    lines.append(cron_line)
    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n", text=True, check=True)

    day_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    console.print(f"\n[green]Schedule installed.[/green]")
    console.print(f"  Runs:    every {day_names[day_num]} at {time}")
    console.print(f"  Command: {binary} run all")
    console.print(f"  Log:     {log_file}")
    console.print(f"\nRun [bold]applypilot schedule --remove[/bold] to uninstall.")


@app.command()
def ready(
    limit: int = typer.Option(25, "--limit", "-l", help="Max jobs to show."),
    min_score: int = typer.Option(7, "--min-score", help="Minimum fit score."),
    open_n: Optional[int] = typer.Option(None, "--open", "-o", help="Open job #N's application URL in your browser."),
) -> None:
    """List jobs with tailored resumes and cover letters ready for manual application."""
    _bootstrap()

    from applypilot.database import get_connection

    conn = get_connection()
    rows = conn.execute(
        """
        SELECT title, company, fit_score, application_url,
               tailored_resume_path, cover_letter_path, url
        FROM jobs
        WHERE tailored_resume_path IS NOT NULL
          AND cover_letter_path IS NOT NULL
          AND applied_at IS NULL
          AND fit_score >= ?
        ORDER BY fit_score DESC, discovered_at DESC
        LIMIT ?
        """,
        (min_score, limit),
    ).fetchall()

    if not rows:
        console.print(
            "[yellow]No jobs ready.[/yellow]\n"
            "Run [bold]applypilot run all[/bold] to discover and prepare applications."
        )
        return

    if open_n is not None:
        idx = open_n - 1
        if 0 <= idx < len(rows):
            import webbrowser
            target = rows[idx]["application_url"] or rows[idx]["url"]
            webbrowser.open(target)
            console.print(f"[green]Opened:[/green] {target}")
        else:
            console.print(f"[red]No job #{open_n}.[/red] Valid range: 1–{len(rows)}")
        return

    console.print(f"\n[bold]{len(rows)} job(s) ready to apply[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("Score", justify="center", width=6)
    table.add_column("Role", min_width=22)
    table.add_column("Company", min_width=16)
    table.add_column("Resume / Cover", min_width=28)
    table.add_column("Apply URL")

    for i, row in enumerate(rows, 1):
        score = row["fit_score"] or 0
        score_color = "green" if score >= 8 else "yellow" if score >= 6 else "red"
        resume = Path(row["tailored_resume_path"]).name if row["tailored_resume_path"] else "—"
        cover = Path(row["cover_letter_path"]).name if row["cover_letter_path"] else "—"
        apply_url = row["application_url"] or row["url"] or "—"

        table.add_row(
            str(i),
            f"[{score_color}]{score}[/{score_color}]",
            row["title"] or "—",
            row["company"] or "—",
            f"R: {resume}\nC: {cover}",
            apply_url,
        )

    console.print(table)
    console.print(
        "\n[dim]Open a job in your browser:  [bold]applypilot ready --open N[/bold][/dim]"
    )
    console.print(
        "[dim]Mark done after submitting:   [bold]applypilot apply --mark-applied URL[/bold][/dim]\n"
    )


if __name__ == "__main__":
    app()
