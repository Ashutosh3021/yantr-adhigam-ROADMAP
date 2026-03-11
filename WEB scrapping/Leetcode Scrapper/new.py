import requests
import time
import json
from bs4 import BeautifulSoup
import re

# ---------- Configuration ----------
LEETCODE_API = "https://leetcode.com/graphql"
HEADERS = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}
DELAY = 1  # seconds between requests to be polite
OUTPUT_FILE = "leetcode_sample_testcases.json"

# GraphQL query to get a list of problems (supports pagination)
PROBLEM_LIST_QUERY = """
query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
  problemsetQuestionList: questionList(
    categorySlug: $categorySlug
    limit: $limit
    skip: $skip
    filters: $filters
  ) {
    total: totalNum
    questions: data {
      titleSlug
      title
      difficulty
      isPaidOnly
    }
  }
}
"""

# GraphQL query to get the HTML content of a problem
QUESTION_CONTENT_QUERY = """
query questionContent($titleSlug: String!) {
  question(titleSlug: $titleSlug) {
    content
  }
}
"""

# ---------- Helper Functions ----------
def fetch_problem_list(limit=50, skip=0):
    """Fetch a page of problems from LeetCode."""
    payload = {
        "operationName": "problemsetQuestionList",
        "query": PROBLEM_LIST_QUERY,
        "variables": {
            "categorySlug": "",
            "limit": limit,
            "skip": skip,
            "filters": {}
        }
    }
    response = requests.post(LEETCODE_API, json=payload, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    return data["data"]["problemsetQuestionList"]

def fetch_problem_content(title_slug):
    """Fetch the HTML content of a problem by its slug."""
    payload = {
        "operationName": "questionContent",
        "query": QUESTION_CONTENT_QUERY,
        "variables": {"titleSlug": title_slug}
    }
    response = requests.post(LEETCODE_API, json=payload, headers=HEADERS)
    response.raise_for_status()
    data = response.json()
    return data["data"]["question"]["content"]

def extract_test_cases_from_html(html):
    """
    Parse the problem HTML and extract sample test cases.
    Returns a list of dicts: [{"input": "...", "output": "..."}, ...]
    """
    soup = BeautifulSoup(html, "html.parser")
    test_cases = []

    # Sample test cases are usually inside <pre> tags, sometimes with <strong>Input:</strong> / <strong>Output:</strong>
    # We'll look for <pre> blocks and try to parse them.
    pre_blocks = soup.find_all("pre")

    for pre in pre_blocks:
        text = pre.get_text()
        # Try to match patterns like:
        # Input: ... Output: ...
        # or "Input: ..." and then later "Output: ..."
        input_match = re.search(r"(?:Input:|Input\s*:?)\s*(.*?)(?=\s*(?:Output:|Output\s*:?|$))", text, re.DOTALL | re.IGNORECASE)
        output_match = re.search(r"(?:Output:|Output\s*:?)\s*(.*?)(?=\s*(?:Explanation:|Constraints:|$))", text, re.DOTALL | re.IGNORECASE)

        if input_match and output_match:
            input_str = input_match.group(1).strip()
            output_str = output_match.group(1).strip()
            test_cases.append({"input": input_str, "output": output_str})
        else:
            # Sometimes it's just two lines: first input, second output
            lines = [line.strip() for line in text.split("\n") if line.strip()]
            if len(lines) >= 2:
                # Heuristic: first line is input, second is output
                test_cases.append({"input": lines[0], "output": lines[1]})

    # If no test cases found via <pre>, try looking for examples in <p> tags (less common)
    if not test_cases:
        paragraphs = soup.find_all("p")
        for p in paragraphs:
            text = p.get_text()
            if "Input:" in text and "Output:" in text:
                # Similar extraction as above
                input_match = re.search(r"Input:\s*(.*?)\s*Output:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
                if input_match:
                    test_cases.append({"input": input_match.group(1).strip(), "output": input_match.group(2).strip()})

    return test_cases

# ---------- Main Scraper ----------
def main():
    all_testcases = {}
    skip = 0
    limit = 50  # number of problems per request
    total_processed = 0

    print("Starting to scrape free LeetCode sample test cases...")

    while True:
        print(f"Fetching problems {skip} to {skip+limit-1}...")
        try:
            page = fetch_problem_list(limit=limit, skip=skip)
            questions = page["questions"]
            total = page["total"]
        except Exception as e:
            print(f"Error fetching problem list: {e}")
            break

        if not questions:
            break

        for q in questions:
            if q["isPaidOnly"]:
                print(f"Skipping premium problem: {q['title']}")
                continue

            print(f"Processing: {q['title']} (slug: {q['titleSlug']})")
            try:
                html_content = fetch_problem_content(q["titleSlug"])
                test_cases = extract_test_cases_from_html(html_content)
                if test_cases:
                    all_testcases[q["titleSlug"]] = {
                        "title": q["title"],
                        "difficulty": q["difficulty"],
                        "testcases": test_cases
                    }
                time.sleep(DELAY)  # be polite
            except Exception as e:
                print(f"  Error processing {q['titleSlug']}: {e}")
                continue

        total_processed += len(questions)
        if total_processed >= total:
            break
        skip += limit

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_testcases, f, indent=2, ensure_ascii=False)

    print(f"\nScraping complete. Saved {len(all_testcases)} problems with sample test cases to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()