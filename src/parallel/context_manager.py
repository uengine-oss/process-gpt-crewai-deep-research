from typing import Any
import openai
import os
import json
from dotenv import load_dotenv
import logging
from contextvars import ContextVar

# 환경 변수 로드 및 OpenAI API 초기화
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

load_dotenv()

# ContextVar 기반으로 crew 실행 컨텍스트 관리
crew_type_var: ContextVar[str] = ContextVar("crew_type", default="unknown")
todo_id_var: ContextVar[str]     = ContextVar("todo_id", default=None)
proc_id_var: ContextVar[str]     = ContextVar("proc_inst_id", default=None)

logger = logging.getLogger(__name__)


def set_crew_context(crew_type: str, todo_id: str = None, proc_inst_id: str = None):
    """
    ContextVar에 crew_type, todo_id, proc_inst_id를 설정하고 토큰을 반환합니다.
    사용 후 reset_crew_context로 복원하세요.
    """
    token_ct  = crew_type_var.set(crew_type)
    token_td  = todo_id_var.set(todo_id)
    token_pid = proc_id_var.set(proc_inst_id)
    return token_ct, token_td, token_pid


def reset_crew_context(token_ct, token_td, token_pid):
    """
    ContextVar 설정을 이전 상태로 복원합니다.
    """
    crew_type_var.reset(token_ct)
    todo_id_var.reset(token_td)
    proc_id_var.reset(token_pid)
    

def summarize(outputs: Any, feedbacks: Any) -> str:
    """주어진 outputs와 feedbacks를 LLM으로 요약 후 결과를 바로 반환합니다."""
    # outputs와 feedbacks를 문자열로 변환
    outputs_str = outputs if isinstance(outputs, str) else json.dumps(outputs, ensure_ascii=False)
    feedbacks_str = feedbacks if isinstance(feedbacks, str) else json.dumps(feedbacks, ensure_ascii=False)
    print("\n\n요약을 위한 LLM호출 시작\n\n")

    # 요약 프롬프트 (outputs와 feedbacks를 함께 처리)
    prompt = f"""새 산출물과 피드백을 병합하여, 아래 형식에 맞는 하나의 통합 요약을 생성하세요.

반드시 지켜야하는 사항들 : 
    1. 결과물 내용이 리포트가 아니라 단순 문자열(텍스트 폼)인 경우, 예 : 사용자 요구사항, 이름, 나이, 피드백 내용 등등... , "목적, 요구사항, 피드백" 만 작성하고, 아래 목차별 핵심 요약은 작성하지 마세요.
    2. 목차는 반드시 결과물 내용에서 추출해야하며, 보고서 형식의 결과물일 때만 진행하고, 단순 문자열인 경우 목차는 작성하지 마세요. 즉, 목차별 핵심 요약을 건너뛰고, 목적, 요구사항, 피드백만 작성하세요.
    3. 피드백은 반드시 피드백 내용에서 추출해야하며, 피드백 내용이 없으면 작성하지 마세요.
    4. 내용이 너무 많아서 2000자가 넘을 경우, 내용을 줄여서라도 2000자 이내로 작성하세요.
    
결과물 내용:
{outputs_str}

피드백 내용:
{feedbacks_str}

===== 요약 형식 (반드시 이 형식을 따르세요) =====

📋 보고서 제목: [정확히 추출한 제목 없으면, 문맥상 흐름을 분석하여 제목을 정의]

📌 목적 : [결과물 내용을 분석하여 목적을 정의]
📌 요구사항 : [결과물 내용을 분석하여 요구사항을 정의]
📌 피드백 : [피드백 내용을 분석하여 피드백을 정의]

👤 작성 정보:
- 작성자: [작성자명]
- 소속부서: [부서명]

🎯 목차별 핵심 요약:

1️⃣ [목차1 제목]:
   • 핵심내용 1: [중요 포인트를 한 문장으로]
   • 핵심내용 2: [주요 데이터나 결과를 한 문장으로]
   • 핵심내용 3: [결론이나 시사점을 한 문장으로]

2️⃣ [목차2 제목]:
   • 핵심내용 1: [중요 포인트를 한 문장으로]
   • 핵심내용 2: [주요 데이터나 결과를 한 문장으로]
   • 핵심내용 3: [결론이나 시사점을 한 문장으로]

3️⃣ [목차3 제목]:
   • 핵심내용 1: [중요 포인트를 한 문장으로]
   • 핵심내용 2: [주요 데이터나 결과를 한 문장으로]
   • 핵심내용 3: [결론이나 시사점을 한 문장으로]

[계속해서 모든 목차에 대해 동일한 형식으로...]

===== 작성 지침 =====
!!중요!! 전체 내용은 2000자 이내로 작성하세요. 
!!중요!! 반드시 보고서 형식이 아닌 결과물일 경우, 목차 없이, 목적, 요구사항만 작성하세요.
1. 목차는 결과물 내용에서 정확히 추출하여 누락 없이 모두 포함
2. 각 목차별로 내용을 요약하여, 핵심내용만 추출 (최대 3줄 이내로 작성)
3. 숫자, 데이터, 구체적 사실을 우선적으로 포함
4. 메타데이터(작성자, 부서 등)는 반드시 찾아서 포함 (없으면 "정보 없음"으로 표시)
5. 구조화된 형식을 정확히 유지
6. 내용 순서는 그대로 유지
"""

    # LLM 호출
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": """당신은 전문적인 요약 전문가입니다. 
                    
주요 역할:
- 복잡한 산출물(보고서, 폼 등)을 구조화된 형식으로 정확히 요약
- 목차별 핵심 내용을 빠짐없이 추출
- 메타데이터와 중요 데이터를 정확히 파악
- 비즈니스 문서의 핵심 가치를 보존하면서 간결하게 정리

작업 원칙:
1. 정확성: 원문의 내용을 왜곡하지 않고 정확히 요약
2. 완전성: 모든 목차와 중요 정보를 누락 없이 포함
3. 구조화: 일관된 형식으로 읽기 쉽게 정리
4. 간결성: 핵심만 추출하여 효율적으로 전달
5. 실용성: 후속 작업에 활용하기 쉬운 형태로 가공"""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=3000,
            temperature=0.1
        )
        summary = response.choices[0].message.content.strip()
        print(f"✅ Context 요약 완료: {len(summary)}자", flush=True)
        return summary
    except Exception as e:
        print(f"❌ Context 요약 실패: {type(e).__name__}: {e}", flush=True)
        raise