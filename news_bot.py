#!/usr/bin/env python3
"""
Telegram News Digest Bot (Israeli edition)
Aggregates Israeli news RSS feeds, filters with Claude AI,
and delivers a Hebrew summary.

Requires Python 3.9+
Env vars: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN
"""

import asyncio
import calendar
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

# Public Telegram channels scraped via t.me/s/ web preview (no MTProto needed)
TELEGRAM_CHANNELS: list[str] = [
    "KanNewsTwitter",
    "amitsegal",
    "N12chat",
    "N12nws",
    "lieldaphna",
    "danielamram3",
    "moriahdoron",
    "MichaelShemesh",
    "grinzaig",
    "BenTzionM",
]

MAX_ITEMS_PER_FEED = 15
MAX_MSGS_PER_CHANNEL = 20
MAX_ITEMS_FOR_CLAUDE = 50  # hard cap sent to Claude to stay under token rate limit
MAX_ITEMS_PER_SOURCE = 4   # max items taken from each individual feed / channel
DESCRIPTION_MAX_LEN = 400
HTTP_TIMEOUT = 15.0
RECENCY_MIN_HOURS = 0   # include stories right up to now
RECENCY_MAX_HOURS = 12  # exclude stories older than this

# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    source: str
    title: str
    description: str
    url: str = ""
    published: datetime | None = field(default=None)


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
            ts = entry.get("published_parsed") or entry.get("updated_parsed")
            if ts:
                published = datetime.fromtimestamp(calendar.timegm(ts), tz=timezone.utc)
                # Some feeds (e.g. walla, mako) publish IDT (UTC+3) without a
                # timezone marker; feedparser then treats it as UTC, making the
                # timestamp appear up to 3 h in the future. Correct it.
                if published > datetime.now(tz=timezone.utc) + timedelta(hours=1):
                    published -= timedelta(hours=3)
            else:
                published = None
            if title:
                items.append(NewsItem(source=name, title=title,
                                      description=description, url=link,
                                      published=published))
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


# ── Telegram channel scraping (t.me/s/ public web preview) ───────────────────

async def _fetch_channel(
    client: httpx.AsyncClient, channel: str
) -> list[NewsItem]:
    url = f"https://t.me/s/{channel}"
    try:
        resp = await client.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            logger.warning("Telegram @%s: HTTP %s", channel, resp.status_code)
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        items: list[NewsItem] = []
        for msg in soup.select(".tgme_widget_message")[-MAX_MSGS_PER_CHANNEL:]:
            text_el = msg.select_one(".tgme_widget_message_text")
            if not text_el:
                continue
            text = text_el.get_text(" ").strip()
            if len(text) < 20:
                continue
            date_el = msg.select_one(".tgme_widget_message_date")
            link = date_el.get("href", "") if date_el else ""
            time_el = msg.select_one(".tgme_widget_message_date time")
            dt_str = time_el.get("datetime", "") if time_el else ""
            try:
                published = datetime.fromisoformat(dt_str) if dt_str else None
            except ValueError:
                published = None
            lines = text.split("\n", 1)
            title = lines[0][:200]
            description = lines[1].strip()[:DESCRIPTION_MAX_LEN] if len(lines) > 1 else ""
            items.append(NewsItem(
                source=f"@{channel}", title=title,
                description=description, url=link, published=published,
            ))
        logger.info("Telegram @%s → %d messages", channel, len(items))
        return items
    except Exception as exc:
        logger.warning("Telegram @%s failed: %s", channel, exc)
        return []


async def fetch_all_channels() -> list[NewsItem]:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        batches = await asyncio.gather(
            *[_fetch_channel(client, ch) for ch in TELEGRAM_CHANNELS]
        )
    return [item for batch in batches for item in batch]


# ── Recency filter ────────────────────────────────────────────────────────────

def filter_recent(items: list[NewsItem]) -> list[NewsItem]:
    now = datetime.now(tz=timezone.utc)
    oldest = now - timedelta(hours=RECENCY_MAX_HOURS)
    newest = now - timedelta(hours=RECENCY_MIN_HOURS)
    result = [
        i for i in items
        if i.published is not None and oldest <= i.published <= newest
    ]
    logger.info(
        "Recency filter: %d → %d items (last %dh, oldest cutoff %s UTC)",
        len(items), len(result), RECENCY_MAX_HOURS, oldest.strftime("%H:%M"),
    )
    return result


# ── Claude input cap ─────────────────────────────────────────────────────────

def cap_for_claude(items: list[NewsItem]) -> list[NewsItem]:
    """Round-robin up to MAX_ITEMS_PER_SOURCE items from each source, capped at MAX_ITEMS_FOR_CLAUDE."""
    by_source: dict[str, list[NewsItem]] = {}
    for item in items:
        by_source.setdefault(item.source, []).append(item)

    result: list[NewsItem] = []
    for round_i in range(MAX_ITEMS_PER_SOURCE):
        for source_items in by_source.values():
            if round_i < len(source_items):
                result.append(source_items[round_i])
            if len(result) >= MAX_ITEMS_FOR_CLAUDE:
                break
        if len(result) >= MAX_ITEMS_FOR_CLAUDE:
            break

    counts = {}
    for item in result:
        counts[item.source] = counts.get(item.source, 0) + 1
    breakdown = "  ".join(f"{s}:{n}" for s, n in sorted(counts.items()))
    logger.info("Claude cap: %d → %d sent  [%s]", len(items), len(result), breakdown)
    return result


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

SYSTEM_PROMPT = """אתה עורך חדשות מנוסה. המשימה שלך היא לעבור על רשימת פריטי המידע שהתקבלו (לפחות 70 פריטים) ולזקק מתוכם עד 13 סיפורים איכותיים שפורסמו ב-8 השעות האחרונות בלבד.

**הנחיות לסינון התוכן:**
- התמקד בסיפורים "מהשטח": עדויות, יוזמות אזרחיות, דוחות חדשים וסיפורים חברתיים/כלכליים.
- סנן החוצה "חדשות קשות" (Hard News): כותרות פוליטיות שגרתיות, הודעות רשמיות של הממשלה/צה"ל, ונושאים שמופיעים בכל אתרי המיינסטרים בו-זמנית.
- חפש את הערך המוסף והזווית הבלעדית או האנושית.
- תעדף חשיפות בתחומים הבאים: חינוך, טכנולוגיה, עמותות, כלכלה, רווחה ושחיתות.

**דוגמאות לסיפורים טובים:**
- "מחדל המיירטים: המלאי הדליל וההחלטה המודעת לא ליירט" (חשיפה, החלטה מוסתרת)
- "1 מכל 5 בני נוער חווים אלימות ומתחמשים להגנה" (סטטיסטיקה מפתיעה שחושפת מגמה)
- "מגיפת החניונים: ילודים נדבקו במחלות, חולי סרטן בתנאים מחפירים" (בעיה מערכתית מוסתרת)
- "טרור הפרוטקשן: חוב של אלפי שקלים שתפח למאות אלפים" (דפוס פשע ספציפי)
- סיפור על עסק/מקום עם זווית אנושית מפתיעה שגורמת לעצור ולחשוב

**דרישות פורמט (Telegram Markdown):**
📋 *דיגסט חדשות איכותי*
_[תאריך ושעה - שעון ישראל]_ 🇮🇱
━━━━━━━━━━━━━━━
*[כותרת הידיעה]*
- **תקציר:** [2-3 משפטים תמציתיים על מהות הסיפור והזווית הייחודית שלו]
- 🕒 **שעת פרסום:** [השעה המקורית של הידיעה]
- 📍 **מקור:** [שם המקור/ערוץ הטלגרם]
- 🔗 **לינק:** [קישור ישיר לידיעה/לפוסט]
🎙️ **מרואיינים מומלצים:**
1. [שם מלא אם מוזכר בכתבה / תפקיד אם השם אינו ידוע] — [קשר לסיפור]
2. [...]
3. [...]
[חזור על המבנה עבור עד 13 סיפורים נבחרים]
━━━━━━━━━━━━━━━
_נסרקו [X] פריטים מה-8 השעות האחרונות_
_נבחרו [Y] סיפורים (מקסימום 13)_

**הנחיות למרואיינים:**
- הצע עד 3 מרואיינים לכל סיפור.
- השתמש בשמות אמיתיים רק אם הם מופיעים בכתבה עצמה; אחרת ציין תפקיד/סוג.
- תעדף מרואיינים לא-מובנים מאליהם: עדי ראייה, מומחים מקומיים, עובדי עמותות, אקדמאים, אזרחים מושפעים, בכירים מוניציפליים.
- אל תציע פוליטיקאים, שרים, דוברי צבא או דמויות ממסדיות שגרתיות.
"""


async def summarize_with_claude(news_items: list[NewsItem]) -> str:
    sections: list[str] = [
        f"תאריך: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "=== פריטי חדשות מאתרי חדשות ישראליים ===",
    ]
    for item in news_items:
        sections.append(f"[{item.source.upper()}] {item.title}")
        if item.published:
            sections.append(f"  time: {item.published.strftime('%H:%M')}")
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
        "הדיגסט סורק 6 אתרי חדשות ישראליים + 10 ערוצי טלגרם של עיתונאים,\n"
        "ומתמקד בסיפורים מהשטח, מצוקה כלכלית, יוזמות אזרחיות ואירועי ביטחון."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*פקודות הבוט:*\n\n"
        "/digest — הפק דיגסט חדשות עכשיו\n"
        "/start — הצג הודעת ברוכים הבאים\n"
        "/help — הצג הודעה זו\n\n"
        "*מקורות RSS:*\n"
        "ynet · walla · mako · מעריב · ישראל היום · המקום\n\n"
        "*ערוצי טלגרם:*\n"
        "@KanNewsTwitter · @amitsegal · @N12chat · @N12nws\n"
        "@lieldaphna · @danielamram3 · @moriahdoron\n"
        "@MichaelShemesh · @grinzaig · @BenTzionM",
        parse_mode="Markdown",
    )


FETCH_TIMEOUT = 120.0   # seconds budget for RSS + Telegram combined
GLOBAL_TIMEOUT = 300.0  # hard cap on the entire /digest operation (5 minutes)


def _format_fallback(items: list[NewsItem]) -> str:
    """Simple bullet list sent when the global timeout fires before Claude responds."""
    lines = [f"⚠️ *תם הזמן — {len(items)} פריטים שנאספו:*\n"]
    for item in items[:15]:
        time_str = f" | {item.published.strftime('%H:%M')}" if item.published else ""
        lines.append(f"• [{item.source}{time_str}] {item.title}")
    return "\n".join(lines)


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    loading = await update.message.reply_text("⏳ (1/3) מאחזר כתבות RSS...")
    collected: list[NewsItem] = []  # populated before Claude; used as fallback on timeout

    async def _run() -> None:
        nonlocal collected

        loop = asyncio.get_event_loop()
        deadline = loop.time() + FETCH_TIMEOUT

        rss_task = asyncio.create_task(fetch_all_rss())
        channel_task = asyncio.create_task(fetch_all_channels())

        rss_items: list[NewsItem] = []
        channel_items: list[NewsItem] = []

        try:
            rss_items = await asyncio.wait_for(
                asyncio.shield(rss_task),
                timeout=max(1.0, deadline - loop.time()),
            )
        except asyncio.TimeoutError:
            rss_task.cancel()
            logger.warning("RSS fetch timed out, continuing with empty list")

        await loading.edit_text(
            f"⏳ (2/3) מאחזר ערוצי טלגרם... (RSS: {len(rss_items)} פריטים)"
        )

        remaining = deadline - loop.time()
        if remaining > 1.0:
            try:
                channel_items = await asyncio.wait_for(
                    channel_task, timeout=remaining,
                )
            except asyncio.TimeoutError:
                channel_task.cancel()
                logger.warning("Telegram fetch timed out, continuing with empty list")
        else:
            channel_task.cancel()
            logger.warning("No time left for Telegram fetch, skipping")

        collected = deduplicate(filter_recent(rss_items + channel_items))
        logger.info("Digest: %d items after filter+dedup", len(collected))

        await loading.edit_text("⏳ (3/3) מסכם עם Claude AI...")

        summary = await summarize_with_claude(cap_for_claude(collected))
        await loading.delete()
        await _reply_safe(update, summary)

    try:
        await asyncio.wait_for(_run(), timeout=GLOBAL_TIMEOUT)
    except asyncio.TimeoutError:
        logger.error("Global digest timeout after %ds", GLOBAL_TIMEOUT)
        if collected:
            try:
                await loading.edit_text(_format_fallback(collected), parse_mode="Markdown")
            except BadRequest:
                await loading.edit_text(_format_fallback(collected))
        else:
            await loading.edit_text("❌ תם הזמן (5 דקות) לפני שנאסף חומר. נסה שוב.")
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
