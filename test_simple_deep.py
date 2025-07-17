import os
from openai import OpenAI

   # 1. 환경 변수에서 API 키 읽기
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 설정되지 않았습니다.")


# 2. 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# 3. 시스템 메시지와 사용자 쿼리 정의
system_message = """
You are a professional researcher preparing a structured, data-driven report on behalf of a global health economics team. Your task is to analyze the health question the user poses.
Do:
- Focus on data-rich insights: include specific figures, trends, statistics, and measurable outcomes (e.g., reduction in hospitalization costs, market size, pricing trends, payer adoption).
- When appropriate, summarize data in a way that could be turned into charts or tables, and call this out in the response (e.g., “this would work well as a bar chart comparing per-patient costs across regions”).
- Prioritize reliable, up-to-date sources: peer-reviewed research, health organizations (e.g., WHO, CDC), regulatory agencies, or pharmaceutical earnings reports.
- Include inline citations and return all source metadata.
Be analytical, avoid generalities, and ensure that each section supports data-backed reasoning that could inform healthcare policy or financial modeling.
"""

user_query = "Research the economic impact of semaglutide on global healthcare systems."

# 4. Deep Research API 호출
response = client.responses.create(
    model="o3-deep-research-2025-06-26",
    input=[
        {
            "role": "developer",
            "content": [
                {
                    "type": "input_text",
                    "text": system_message,
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_query,
                }
            ]
        }
    ],
    reasoning={"summary": "auto"},
    tools=[
        {"type": "web_search_preview"},
        {
            "type": "code_interpreter",
            "container": {"type": "auto", "file_ids": []}
        }
    ]
)

# 5. 최종 보고서 텍스트 출력
print(response.output[-1].content[0].text)
