#!/usr/bin/env python3
"""
Parallel Polling Deep Research with Background Mode (Preview)

- Requires: pip install requests
- Set env: export OPENAI_API_KEY="YOUR_KEY"
"""

import os, time, threading, requests

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("OPENAI_API_KEY 환경 변수를 설정하세요")

HEADERS = {
    "Authorization":       f"Bearer {API_KEY}",
    "Content-Type":        "application/json",
    "OpenAI-API-Version":  "2025-06-26"        # 🔑 Preview 헤더
}

prompts = [
    "시장 동향 분석에 대한 심층 리서치 결과를 작성하세요.",
    "전체 요약 및 최종 제안을 작성하세요."
]

results = {}

def start_request(prompt: str) -> str:
    payload = {
        "model":     "o3-deep-research",  # 🔑 OpenAI-API-Version 헤더를 활용하도록 모델 ID 수정
        "input":     [
            {"role":"developer","content":[{"type":"input_text","text":"You are a research assistant."}]},
            {"role":"user",     "content":[{"type":"input_text","text":prompt}]}
        ],
        "tools":     [{"type":"web_search_preview"}],
        "reasoning": {"summary":"auto"},
        "background": True
    }
    r = requests.post("https://api.openai.com/v1/responses",
                      headers=HEADERS, json=payload)
    r.raise_for_status()
    data = r.json()
    rid = data["id"]
    results[rid] = {"prompt": prompt, "status": data["status"], "output": None}
    print(f"[{rid[:8]}] 요청 생성 – 상태: {data['status']}")
    return rid

def poll_request(rid: str, interval: int = 5):
    while True:
        time.sleep(interval)
        r = requests.get(f"https://api.openai.com/v1/responses/{rid}",
                         headers=HEADERS)
        r.raise_for_status()
        d = r.json()
        results[rid]["status"] = d["status"]
        txt = d.get("output", {}).get("text", "")
        if txt and txt != results[rid].get("output"):
            results[rid]["output"] = txt
            print(f"[{rid[:8]}] 중간 길이: {len(txt)}")
        if d["status"] in ("queued","in_progress"):
            continue
        if d["status"] == "succeeded":
            print(f"[{rid[:8]}] 완료! 최종 길이: {len(d['output']['text'])}")
        else:
            print(f"[{rid[:8]}] 실패: {d.get('error')}")
        break

def main():
    threads = []
    for p in prompts:
        rid = start_request(p)
        t = threading.Thread(target=poll_request, args=(rid,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print("\n# 통합 리포트\n")
    for rid, info in results.items():
        print(f"## [{rid[:8]}] {info['prompt']}")
        print(info["output"] or "_결과 없음_")
        print()

if __name__ == "__main__":
    main()
