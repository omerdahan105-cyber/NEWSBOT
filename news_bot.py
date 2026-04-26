#!/usr/bin/env python3
"""
Telegram News Digest Bot (Israeli edition)
Aggregates Israeli news RSS feeds, filters with Claude AI,
and delivers a Hebrew summary.

Requires Python 3.9+
Env vars: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN
"""

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher

import anthropic
import feedparser
import httpx
from bs4 import BeautifulSoup
from telegram import Update
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CLAUDE_MODEL = "claude-opus-4-7"

# Israeli news RSS feeds
RSS_FEEDS: dict[str, str] = {
    "ynet":        "https://www.ynet.co.il/Integration/StoryRss2.xml",
    "walla":       "https://rss.walla.co.il/feed/1",
    "mako":        "https://www.mako.co.il/rss/news-military.xml",
    "maariv":      "https://www.maariv.co.il/rss/rssfeedsTech.aspx",
    "israelhayom": "https://www.israelhayom.co.il/rss.xml",
    "ha-makom":    "https://www.ha-makom.co.il/feed",
}

MAX_ITEMS_PER_FEED = 15
DESCRIPTION_MAX_LEN = 400
HTTP_TIMEOUT = 15.0

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    source: str
    title: str
    description: str
    url: str = ""


# ── RSS fetching ──────────────────────────────────────────────────────────────

async def _fetch_feed(
    client: httpx.AsyncClient, name: str, url: str
) -> list[NewsItem]:
    try:
        resp = await client.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        # feedparser is CPU-bound; keep off the event loop
        feed = await asyncio.to_thread(feedparser.parse, resp.text)
        items: list[NewsItem] = []
        for entry in feed.entries[:MAX_ITEMS_PER_FEED]:
            title = (entry.get("title") or "").strip()
            raw_desc = entry.get("summary") or entry.get("description") or ""
            description = BeautifulSoup(raw_desc, "html.parser").get_text(" ").strip()
            description = description[:DESCRIPTION_MAX_LEN]
            link = entry.get("link") or ""
            if title:
                items.append(NewsItem(source=name, title=title,
                                      description=description, url=link))
        logger.info("RSS %s → %d items", name, len(items))
        return items
    except Exception as exc:
        logger.warning("RSS %s failed: %s", name, exc)
        return []


async def fetch_all_rss() -> list[NewsItem]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_feed(client, name, url) for name, url in RSS_FEEDS.items()]
        )
    return [item for batch in batches for item in batch]


# ── Deduplication ─────────────────────────────────────────────────────────────

def _normalize_title(title: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', title)).strip()

def deduplicate(items: list[NewsItem]) -> list[NewsItem]:
    seen: list[str] = []
    result: list[NewsItem] = []
    for item in items:
        key = _normalize_title(item.title)
        if any(SequenceMatcher(None, key, s).ratio() > 0.7 for s in seen):
            continue
        seen.append(key)
        result.append(item)
    logger.info("Dedup: %d → %d items", len(items), len(result))
    return result


# ── Claude summarization ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """אתה עורך חדשותי מנוסה שמכין דיגסט יומי בעברית.

**משימה:** לסנן מתוך רשימת פריטי חדשות ומקורות מגוונים ולהפיק דיגסט קצר ואיכותי.

**סוגי סיפורים שאתה מחפש:**
- סיפורים בלעדיים וחקירות עיתונאיות
- עדויות ישירות מהשטח, דיווחים של אזרחים רגילים
- מצוקה כלכלית — יוקר המחיה, עסקים קטנים, פרנסה, מחירים
- אירועי ביטחון שדווחו על ידי תושבים ועדי ראייה
- בעיות עירוניות ומוניציפליות שמשפיעות על תושבים
- מחאות חברתיות, הפגנות, עצומות
- סיפורים אנושיים שנקברים מתחת לכותרות הגדולות
- חשיפות, סקופים, פרטים שלא ברורים מהסיקור המיינסטרים
- יוזמות אזרחיות חיוביות ונחמדות
- סיפורים אנושיים מעוררי השראה מהשטח
- פעילות קהילתית וחברתית

**מה להדיר:**
- כותרות שגרתיות שכולם כבר יודעים (הצהרות פוליטיות רגילות, הודעות ממשלה שגרתיות)
- אופנה, גוסיפ, בידור, ספורט
- ידיעות PR ופרסומות סמויות
- כפילויות — אם אותו סיפור מופיע ממספר מקורות, בחר הגרסה הטובה ביותר

**פורמט הפלט (Telegram Markdown):**

📋 *דיגסט חדשות ישראל*
_[תאריך וזמן]_

━━━━━━━━━━━━━━━

[כותרת קצרה ומשכנעת](url)
[2-3 משפטים תמציתיים המספרים את הסיפור]
📍 *מקור:* [שם המקור המדויק]

[חזור על הפורמט לכל סיפור]

━━━━━━━━━━━━━━━
_נסרקו [X] פריטים · נבחרו [Y] סיפורים_

**הנחיות:**
- בחר 5–8 סיפורים בלבד — רק הטובים ביותר
- תעדף תוכן שמגיע מהשטח ומאזרחים
- כתוב בעברית ברורה, ישירה, ולא עיתונאית-שגרתית
- **חובה:** כל סיפור חייב לכלול את שם המקור המדויק בשדה 📍 *מקור:*
- **חובה:** אם לפריט יש url, הפוך את הכותרת לקישור לחיץ בפורמט Telegram: [כותרת](url)
- אם אין סיפורים מעניינים — ציין זאת בכנות"""


async def summarize_with_claude(news_items: list[NewsItem]) -> str:
    sections: list[str] = [
        f"תאריך: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "=== פריטי חדשות מאתרי חדשות ישראליים ===",
    ]
    for item in news_items:
        sections.append(f"[{item.source.upper()}] {item.title}")
        if item.url:
            sections.append(f"  url: {item.url}")
        if item.description:
            sections.append(f"  {item.description}")
        sections.append("")

    raw_content = "\n".join(sections)
    total_items = len(news_items)

    client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    message = await client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"סרוק {total_items} פריטים חדשותיים והפק דיגסט:\n\n"
                    f"{raw_content}"
                ),
            }
        ],
    )
    return message.content[0].text


# ── Message helpers ───────────────────────────────────────────────────────────

def _split_message(text: str, max_len: int = 4000) -> list[str]:
    """Split on paragraph boundaries to preserve Markdown structure."""
    if len(text) <= max_len:
        return [text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len = 0

    for para in text.split("\n\n"):
        para_len = len(para) + 2
        if current_len + para_len > max_len and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(para)
        current_len += para_len

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


async def _reply_safe(update: Update, text: str) -> None:
    """Send reply with Markdown; fall back to plain text if Telegram rejects it."""
    for chunk in _split_message(text):
        try:
            await update.message.reply_text(chunk, parse_mode="Markdown")
        except BadRequest:
            await update.message.reply_text(chunk)


# ── Telegram command handlers ─────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "שלום! אני בוט דיגסט חדשות ישראל 📰\n\n"
        "שלח /digest לקבלת סיכום חדשות מותאם אישית בעברית.\n\n"
        "הדיגסט סורק 6 אתרי חדשות ישראליים ומתמקד בסיפורים מהשטח,\n"
        "מצוקה כלכלית, יוזמות אזרחיות, פעילות קהילתית ואירועי ביטחון."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*פקודות הבוט:*\n\n"
        "/digest — הפק דיגסט חדשות עכשיו\n"
        "/start — הצג הודעת ברוכים הבאים\n"
        "/help — הצג הודעה זו\n\n"
        "*מקורות:*\n"
        "ynet · walla · mako · מעריב · ישראל היום · המקום",
        parse_mode="Markdown",
    )


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    loading = await update.message.reply_text(
        "⏳ אוסף חדשות מ-6 מקורות ישראליים...\nכ-15 שניות"
    )

    try:
        news_items = deduplicate(await fetch_all_rss())
        logger.info("Digest: %d items after dedup", len(news_items))

        summary = await summarize_with_claude(news_items)

        await loading.delete()
        await _reply_safe(update, summary)

    except anthropic.APIError as exc:
        logger.error("Claude API error: %s", exc)
        await loading.edit_text("❌ שגיאה בשירות Claude. בדוק את מפתח ה-API.")
    except Exception as exc:
        logger.error("Digest failed: %s", exc, exc_info=True)
        await loading.edit_text("❌ אירעה שגיאה. נסה שוב מאוחר יותר.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    if not ANTHROPIC_API_KEY:
        raise SystemExit("ANTHROPIC_API_KEY environment variable is not set")
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN environment variable is not set")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("digest", cmd_digest))

    logger.info("Bot starting — model: %s", CLAUDE_MODEL)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
