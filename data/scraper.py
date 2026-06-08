"""
Script scraping data dari GitHub Issues dan Reddit
Jalankan di Colab/Kaggle dengan:
  pip install praw PyGithub pandas
"""
import os
import json
import time
import pandas as pd
from datetime import datetime

# GitHub Issues Scraper
def scrape_github_issues(
    token: str,
    repos: list,
    keywords: list,
    max_per_keyword: int = 30,
    output_path: str     = "data/raw/scraped_github.csv",
):
    """
    Scraping GitHub Issues via Search API (jauh lebih cepat dari iterate all issues).
    Pakai g.search_issues() agar filtering dilakukan server-side.
    """
    try:
        from github import Github, Auth
    except ImportError:
        print("Install dulu: pip install PyGithub")
        return

    g       = Github(auth=Auth.Token(token))
    records = []
    seen    = set()  # dedup by url

    for repo_name in repos:
        print(f"Scraping {repo_name}...")
        for keyword in keywords:
            try:
                query  = f'"{keyword}" repo:{repo_name} type:issue'
                issues = g.search_issues(query=query, sort="updated", order="desc")

                count = 0
                for issue in issues:
                    if count >= max_per_keyword:
                        break
                    if issue.html_url in seen:
                        continue
                    seen.add(issue.html_url)

                    body = issue.body or ""
                    records.append({
                        "source":     "github",
                        "repo":       repo_name,
                        "keyword":    keyword,
                        "title":      issue.title,
                        "body":       body[:2000],
                        "tool_calls": extract_tool_calls_from_text(issue.title + " " + body),
                        "url":        issue.html_url,
                        "created_at": str(issue.created_at),
                    })
                    count += 1
                    time.sleep(0.3)

                print(f"  [{keyword}]: {count} issues")
                time.sleep(1)  # hindari secondary rate limit search API

            except Exception as e:
                print(f"  Error [{keyword}] di {repo_name}: {e}")
                time.sleep(2)

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"GitHub: {len(df)} records → {output_path}")
    return df


# Reddit Scraper
def scrape_reddit(
    client_id:     str,
    client_secret: str,
    user_agent:    str,
    subreddits:    list,
    keywords:      list,
    limit:         int = 500,
    output_path:   str = "data/raw/scraped_reddit.csv",
):
    """
    Scraping Reddit posts dari subreddit terkait LLM agent jailbreak.
    """
    try:
        import praw
    except ImportError:
        print("Install dulu: pip install praw")
        return

    reddit  = praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)
    records = []

    for sub_name in subreddits:
        print(f"Scraping r/{sub_name}...")
        try:
            sub = reddit.subreddit(sub_name)

            for keyword in keywords:
                results = sub.search(keyword, limit=limit, sort="relevance", time_filter="year")

                for post in results:
                    body = post.selftext or ""
                    if len(body) < 50:
                        continue

                    records.append({
                        "source":     "reddit",
                        "subreddit":  sub_name,
                        "keyword":    keyword,
                        "title":      post.title,
                        "body":       body[:2000],
                        "tool_calls": extract_tool_calls_from_text(body),
                        "url":        f"https://reddit.com{post.permalink}",
                        "created_at": str(datetime.fromtimestamp(post.created_utc)),
                    })
                    time.sleep(0.3)

        except Exception as e:
            print(f"  Error r/{sub_name}: {e}")

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Reddit: {len(df)} records → {output_path}")
    return df


# Ekstrak pola tool call dari teks
def extract_tool_calls_from_text(text: str) -> str:
    """
    Ekstrak pola tool call dari teks diskusi.
    Mencari pola seperti: function_name(args) atau tool_call JSON.
    """
    import re
    patterns = [
        r'\b\w+_\w+\([^)]{0,200}\)',         # pola fungsi: nama_func(args)
        r'"name"\s*:\s*"([^"]+)"',            # JSON "name": "..."
        r'"function"\s*:\s*"([^"]+)"',        # JSON "function": "..."
        r'`([a-z_]+)\([^`]*\)`',              # inline code: `func(args)`
    ]
    found = []
    for pat in patterns:
        found.extend(re.findall(pat, text, re.IGNORECASE))

    # format sebagai JSON string
    tool_calls = [{"name": f, "params": {}} for f in found[:10]]
    return json.dumps(tool_calls)


# Konfigurasi scraping
GITHUB_REPOS = [
    "langchain-ai/langchain",
    "microsoft/autogen",
    "joaomdmoura/crewAI",
    "openai/openai-python",
    "BerriAI/litellm",
]

GITHUB_KEYWORDS = [
    "prompt injection",
    "jailbreak",
    "tool call safety",
    "function call exploit",
    "agent security",
]

REDDIT_SUBREDDITS = [
    "LocalLLaMA",
    "ChatGPT",
    "PromptEngineering",
    "MachineLearning",
]

REDDIT_KEYWORDS = [
    "jailbreak tool call",
    "prompt injection agent",
    "bypass function calling",
    "LLM agent attack",
    "tool use exploit",
]


# Jalankan scraping
if __name__ == "__main__":
    print("=== TCSSC Data Scraper ===")
    print("Isi credential di bawah sebelum menjalankan.\n")

    GITHUB_TOKEN  = os.environ.get("GITHUB_TOKEN", "YOUR_GITHUB_TOKEN")
    REDDIT_ID     = "YOUR_REDDIT_CLIENT_ID"
    REDDIT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"

    df_github = scrape_github_issues(
        token       = GITHUB_TOKEN,
        repos       = GITHUB_REPOS,
        keywords    = GITHUB_KEYWORDS,
        output_path = "data/raw/scraped_github.csv",
    )

    df_reddit = scrape_reddit(
        client_id     = REDDIT_ID,
        client_secret = REDDIT_SECRET,
        user_agent    = "tcssc_research_scraper/1.0",
        subreddits    = REDDIT_SUBREDDITS,
        keywords      = REDDIT_KEYWORDS,
        output_path   = "data/raw/scraped_reddit.csv",
    )

    total = (len(df_github) if df_github is not None else 0) + \
            (len(df_reddit) if df_reddit is not None else 0)
    print(f"\nTotal data scraping: {total} rows")
    print("Scraping selesai. Jalankan pseudolabeler.py untuk melabeli data.")
