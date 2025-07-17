#!/usr/bin/env python3
"""
Asyncio Parallel Streaming Deep Research Report Generator

Requirements:
  - Python 3.8+
  - pip install --upgrade openai

Set environment variable:
  export OPENAI_API_KEY="YOUR_API_KEY"
"""

import os
import asyncio
import openai
from datetime import datetime

# ─────────────────────────────────────────────
# 0) 환경 변수 & 클라이언트 초기화
# ─────────────────────────────────────────────
openai.api_key = os.getenv("OPENAI_API_KEY") or ""
if not openai.api_key:
    raise RuntimeError("ERROR: OPENAI_API_KEY를 설정하세요.")

client = openai.OpenAI()  # async 지원 클라이언트 인스턴스

# ─────────────────────────────────────────────
# 1) 섹션 정의
# ─────────────────────────────────────────────
sections = [
    ("시장 동향 분석",     "시장 동향 분석에 대한 심층 리서치 결과를 작성하세요."),
    ("경쟁사 비교",       "주요 경쟁사 비교 분석을 수행하세요."),
    ("기술적 접근 방법",   "해당 주제의 기술적 접근 방법을 설명하세요."),
    ("구현 아키텍처 설계", "적합한 구현 아키텍처 설계안을 제안하세요."),
    ("비용 및 ROI 분석",   "비용 분석 및 예상 ROI를 계산하세요."),
    ("결론 및 제안",      "전체 요약 및 최종 제안을 작성하세요.")
]

# ─────────────────────────────────────────────
# 2) 툴 설정 (file_search 제거)
# ─────────────────────────────────────────────
tools = [
    {"type": "web_search_preview"},
    {"type": "code_interpreter", "container": {"type": "auto"}}
]

# ─────────────────────────────────────────────
# 3) 이벤트 저장소 (task_id → 리스트 of dict)
# ─────────────────────────────────────────────
events_store: dict[int, list[dict]] = {}

async def run_stream(task_id: int, title: str, prompt: str):
    # 3-1) 저장소 초기화
    events_store[task_id] = []

    try:
        # 3-2) 스트림 열기 (client.responses.create 사용)
        stream = await client.responses.create(
            model="o3-deep-research",
            input=[
                {"role":"system","content":[{"type":"input_text","text":"You are a research assistant."}]},
                {"role":"user",  "content":[{"type":"input_text","text":prompt}]}
            ],
            tools=tools,
            reasoning={"summary":"auto"},
            stream=True,
        )

        # 3-3) 이벤트 순회
        async for evt in stream:
            etype = evt.get("type", "")
            data = evt.get("data", {})
            # 시간·타입·페이로드 기록
            evt_record = {
                "time": datetime.utcnow().isoformat(),
                "type": etype,
                **data
            }
            events_store[task_id].append(evt_record)

            # 콘솔에 이벤트별 출력
            if etype == "response.tool_start":
                print(f"[{task_id}] 🔧 tool_start →", data.get("tool"))
            elif etype in ("response.tool_response", "response.tool_output"):
                out = data.get("output", "")
                print(f"[{task_id}] 📥 tool_response →", str(out)[:100].replace("\n"," "))
            elif etype == "response.output_text.delta":
                print(f"[{task_id}] ✉️", data.get("delta",""), end="", flush=True)
            elif etype == "response.message_end":
                print(f"\n[{task_id}] ✅ message_end")
            else:
                # reasoning, plan, error 등
                print(f"[{task_id}] 📌 {etype}")

    except Exception as e:
        print(f"[{task_id}] ❌ Error during stream:", e)

    finally:
        # 3-4) 스트림 닫기
        try:
            await stream.aclose()
        except:
            pass
        count = len(events_store.get(task_id, []))
        print(f"[{task_id}] {title} 완료, 이벤트 수집: {count}개")
        return task_id, title

async def main():
    # 4) create_task + gather 로 병렬 실행
    tasks = [
        asyncio.create_task(run_stream(i+1, sec[0], sec[1]))
        for i, sec in enumerate(sections)
    ]
    results = await asyncio.gather(*tasks)

    # 5) 각 task 의 이벤트 히스토리 출력
    for task_id, title in sorted(results):
        print(f"\n=== Task {task_id}: {title} 이벤트 히스토리 ===")
        for e in events_store.get(task_id, []):
            print(e)

if __name__ == "__main__":
    asyncio.run(main())
