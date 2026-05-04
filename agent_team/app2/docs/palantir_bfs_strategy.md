# BFS Scraping Strategy: Palantir Wikipedia Link Network

## 1. Overview
This document outlines a strategy for crawling and mapping the network of references and internal links starting from the [Palantir Technologies Wikipedia page](https://en.wikipedia.org/wiki/Palantir_Technologies). The goal is to perform a Breadth-First Search (BFS) crawl to a depth of 2.

## 2. Crawl Architecture: BFS (Depth 2)

### Constraints
- **Maximum Sites:** Total URLs scraped will not exceed 100.
- **Depth:** Maximum depth of 2.

### Level 0: Seed
- **Starting Point:** `https://en.wikipedia.org/wiki/Palantir_Technologies`

### Level 1: Direct Links & References
- **Action:** Extract all unique outgoing URLs from the seed page.
- **Targets:**
    - **Internal Wiki Links:** Links to other Wikipedia articles (e.g., `/wiki/Peter_Thiel`).
    - **External References:** URLs found in the "References" and "External links" sections.
- **Queue:** All extracted URLs are added to the Level 1 queue.

### Level 2: Second Degree Connections
- **Action:** Visit URLs gathered in Level 1 until the 100-site limit is reached.
- **Goal:** Extract URLs from these pages to understand the broader context or citation network.
- **Filter:** To prevent infinite growth, we apply strict filtering (e.g., only English Wikipedia or specific domains).

## 3. Algorithm Implementation (Pseudocode)

```python
queue = [(seed_url, 0)]  # (url, depth)
visited = set()
results = []
MAX_SITES = 100

while queue and len(visited) < MAX_SITES:
    current_url, depth = queue.pop(0)
    
    if depth > 2 or current_url in visited:
        continue
    
    visited.add(current_url)
    html = fetch_with_politeness(current_url)
    links = extract_links(html)
    
    results.append({
        "source": current_url,
        "depth": depth,
        "links_found": len(links)
    })
    
    for link in links:
        if depth + 1 <= 2:
            queue.append((link, depth + 1))
```

## 4. Technical Specifications

### Data Extraction Fields
- **Source URL:** The page being scraped.
- **Link Type:** Categorized as `internal` (Wikipedia) or `external` (External source).
- **Link Text:** The anchor text or citation title.
- **Crawl Depth:** 0, 1, or 2.

### Constraints & Politeness
- **Max Total Sites:** 100
- **User-Agent:** Descriptive header (e.g., `PalantirNetworkBot/1.0 (contact: user@example.com)`)
- **Rate Limiting:** 2-3 second delay between requests.
- **Robots.txt:** Strict adherence to Wikipedia's crawl-delay and disallowed paths.
- **Concurrent Requests:** Limited to 1 thread to avoid overwhelming the target server.

## 5. Risk Assessment
- **Explosion of Nodes:** A single Wikipedia page can have 500+ links. A Depth 2 crawl could easily result in 250,000+ requests.
- **Mitigation:** The 100-site limit ensures the crawl remains manageable and respects target resources.
- **Anti-Bot:** Even Wikipedia will throttle or block high-volume BFS crawls without an API key or extremely conservative timing.

## 6. Output Format
Data should be saved as a graph-ready JSON:
```json
{
  "nodes": [{"id": "url1", "depth": 0}, {"id": "url2", "depth": 1}],
  "edges": [{"source": "url1", "target": "url2"}]
}
```
