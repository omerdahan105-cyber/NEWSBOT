#!/usr/bin/env python3
"""
Telegram News Digest Bot (Israeli edition)
Aggregates Israeli news RSS feeds + Hebrew Twitter/X keyword searches,
filters with Claude AI, and delivers a Hebrew summary.

Requires Python 3.9+
Env vars: ANTHROPIC_API_KEY, TELEGRAM_BOT_TOKEN
"""

import asyncio
import logging
import os
import urllib.parse
from dataclasses import dataclass
from datetime import datetime

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
    "ynet":         "https://www.ynet.co.il/Integration/StoryRss2.xml",
    "walla":        "https://rss.walla.co.il/feed/1",
    "haaretz":      "https://www.haaretz.co.il/srv/rss",
    "mako":         "https://rss.mako.co.il/rss/News-n.xml",
    "n12":          "https://www.n12.co.il/rss/all.xml",
    "maariv":       "https://www.maariv.co.il/rss/rssfeedfront.aspx",
    "israelhayom":  "https://www.israelhayom.co.il/Rss.aspx",
    "calcalist":    "https://www.calcalist.co.il/Rss.aspx",
    "globes":       "https://www.globes.co.il/news/rss.aspx",
}

# Nitter instances (open-source Twitter frontend, no auth needed)
NITTER_INSTANCES: list[str] = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.net",
    "https://nitter.catsarch.com",
]

# Hebrew keyword searches — broad citizen / ground-level coverage
SEARCH_QUERIES: list[str] = [
    "יוקר המחיה",          # cost of living
    "מצוקה כלכלית",         # economic hardship
    "אירוע ביטחוני",         # security incident
    "הפגנה",                # protest
    "עירייה בעיה",           # municipality problem
    "תושבים מדווחים",        # residents report
    "עוני ישראל",            # poverty in Israel
    "מחאה חברתית",           # social protest
    "תאונה עדות",            # accident eyewitness
    "שריפה תושבים",          # fire residents
]

# A handful of established Israeli journalists/commentators as bonus signal
JOURNALIST_ACCOUNTS: list[str] = [
    "Ben_Kaspit",
    "raviv_drucker",
    "BarakRavid",
    "YoavLimor",
    "AlonPinkas",
    "Amirhetsroni",
]

MAX_ITEMS_PER_FEED = 15
MAX_RESULTS_PER_SEARCH = 5
MAX_TWEETS_PER_ACCOUNT = 3
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


# ── Nitter / Twitter scraping ─────────────────────────────────────────────────

def _is_hebrew(text: str) -> bool:
    """Return True if the text contains a reasonable amount of Hebrew."""
    hebrew_chars = sum(1 for c in text if "א" <= c <= "ת")
    return hebrew_chars >= 5


def _extract_tweets_from_html(html: str, label: str) -> list[str]:
    """Parse nitter HTML and return tweet texts."""
    soup = BeautifulSoup(html, "html.parser")
    tweets: list[str] = []
    # nitter uses .tweet-content or .tweet-body depending on instance
    for el in soup.select(".tweet-content, .tweet-body"):
        text = el.get_text(" ").strip()
        if len(text) >= 25:
            tweets.append(f"[{label}] {text[:300]}")
    return tweets


async def _search_nitter(
    client: httpx.AsyncClient, query: str, instance: str
) -> list[str]:
    """Search a nitter instance for tweets matching a Hebrew query."""
    encoded = urllib.parse.quote(query)
    url = f"{instance}/search?f=tweets&q={encoded}&lang=iw"
    try:
        resp = await client.get(url, timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return []
        results = _extract_tweets_from_html(resp.text, f"חיפוש: {query}")
        # Keep only results that actually contain Hebrew
        hebrew = [t for t in results if _is_hebrew(t)]
        return hebrew[:MAX_RESULTS_PER_SEARCH]
    except Exception:
        return []


async def _scrape_account(
    client: httpx.AsyncClient, account: str, instance: str
) -> list[str]:
    """Scrape recent tweets from a specific account via nitter."""
    try:
        resp = await client.get(f"{instance}/{account}", timeout=HTTP_TIMEOUT)
        if resp.status_code != 200:
            return []
        results = _extract_tweets_from_html(resp.text, f"@{account}")
        return results[:MAX_TWEETS_PER_ACCOUNT]
    except Exception:
        return []


async def _try_instances(coro_factory) -> list[str]:
    """Try each nitter instance until one succeeds."""
    for instance in NITTER_INSTANCES:
        try:
            result = await coro_factory(instance)
            if result:
                return result
        except Exception:
            continue
    return []


async def fetch_all_twitter() -> list[str]:
    """
    Two-pronged approach:
      1. Keyword searches in Hebrew (citizen / ground-level content)
      2. A handful of Israeli journalist accounts
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsDigestBot/1.0)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:

        # Keyword searches — each query tried on all instances until one responds
        search_tasks = [
            _try_instances(lambda inst, q=query: _search_nitter(client, q, inst))
            for query in SEARCH_QUERIES
        ]

        # Journalist accounts
        account_tasks = [
            _try_instances(lambda inst, a=account: _scrape_account(client, a, inst))
            for account in JOURNALIST_ACCOUNTS
        ]

        all_results = await asyncio.gather(*search_tasks, *account_tasks)

    tweets: list[str] = []
    for batch in all_results:
        tweets.extend(batch)

    logger.info("Twitter → %d raw tweets/posts collected", len(tweets))
    return tweets


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

**מה להדיר:**
- כותרות שגרתיות שכולם כבר יודעים (הצהרות פוליטיות רגילות, הודעות ממשלה שגרתיות)
- אופנה, גוסיפ, בידור, ספורט
- ידיעות PR ופרסומות סמויות
- כפילויות — אם אותו סיפור מופיע ממספר מקורות, בחר הגרסה הטובה ביותר

**פורמט הפלט (Telegram Markdown):**

📋 *דיגסט חדשות ישראל*
_[תאריך וזמן]_

━━━━━━━━━━━━━━━

*[כותרת קצרה ומשכנעת]*
[2-3 משפטים תמציתיים המספרים את הסיפור]
📍 *מקור:* [שם המקור]

[חזור על הפורמט לכל סיפור]

━━━━━━━━━━━━━━━
_נסרקו [X] פריטים · נבחרו [Y] סיפורים_

**הנחיות:**
- בחר 5–8 סיפורים בלבד — רק הטובים ביותר
- תעדף תוכן שמגיע מהשטח ומאזרחים
- כתוב בעברית ברורה, ישירה, ולא עיתונאית-שגרתית
- אם אין סיפורים מעניינים — ציין זאת בכנות"""


async def summarize_with_claude(
    news_items: list[NewsItem], tweets: list[str]
) -> str:
    sections: list[str] = [
        f"תאריך: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "",
        "=== פריטי RSS מאתרי חדשות ישראליים ===",
    ]
    for item in news_items:
        sections.append(f"[{item.source.upper()}] {item.title}")
        if item.description:
            sections.append(f"  {item.description}")
        sections.append("")

    if tweets:
        sections += [
            "",
            "=== ציוצים ופוסטים מטוויטר/X (חיפוש עברית + עיתונאים) ===",
        ]
        sections.extend(tweets)

    raw_content = "\n".join(sections)
    total_items = len(news_items) + len(tweets)

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
        "הדיגסט מסרוק 9 אתרי חדשות ישראליים + חיפוש עברי בטוויטר/X,\n"
        "ומתמקד בסיפורים מהשטח, מצוקה כלכלית, אירועי ביטחון ובעיות אזרחיות."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "*פקודות הבוט:*\n\n"
        "/digest — הפק דיגסט חדשות עכשיו\n"
        "/start — הצג הודעת ברוכים הבאים\n"
        "/help — הצג הודעה זו\n\n"
        "*מקורות:*\n"
        "ynet · walla · haaretz · mako · n12 · מעריב · ישראל היום · כלכליסט · גלובס\n"
        "וחיפוש עברי בטוויטר/X עבור תוכן מהשטח",
        parse_mode="Markdown",
    )


async def cmd_digest(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    loading = await update.message.reply_text(
        "⏳ אוסף חדשות מ-9 מקורות ישראליים + טוויטר...\nכ-30 שניות"
    )

    try:
        news_items, tweets = await asyncio.gather(
            fetch_all_rss(),
            fetch_all_twitter(),
        )
        logger.info(
            "Digest: %d RSS items, %d tweets", len(news_items), len(tweets)
        )

        summary = await summarize_with_claude(news_items, tweets)

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
