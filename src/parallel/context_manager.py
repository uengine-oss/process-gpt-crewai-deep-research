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

**핵심 원칙:**
1. **보고서 형식**: 목차가 있으면 → 목적+요구사항+피드백+목차별 요약
2. **단순 텍스트**: 목차 없으면 → 목적+요구사항
3. **피드백**: 피드백이 있으면 → 피드백 내용 포함
3. **목차 원본 유지**: 있는 그대로 추출, 왜곡 금지 (가장 중요!)
4. **피드백 구분**: 특정 에이전트 지정(@@에이전트명) vs 전역적 피드백 구분
5. **분량 제한**: 전체 2000자 이내
6. **수치 우선**: 숫자, 데이터, 구체적 사실 반드시 포함
    
📌 피드백 분류 기준:
- 특정 에이전트 지정: "@@에이전트명은 이렇게 해라" → 해당 에이전트명 명시
- 전역적 피드백: "이렇게 수정해라", "더 자세히 써라" → 전역적으로 분류

⚠️ 필수: 
목차명는 원본 그대로, 수치 데이터 우선, 2000자 이내 (목차 명 왜곡 및 수정 금지지)
없는 목차를 피드백을 보고 생성하지말고, 있는 그대로 목차를 추출하고, 피드백은 같이 전달되니까 현재 단계에서 반영하지 않아도 됩니다.
결과물은 반드시 그저 요약 및 핵심 정보 추출일 뿐입니다. 피드백을 반영하는게 아닙니다.

**결과물 내용:** {outputs_str}
**피드백 내용:** {feedbacks_str}

===== 요약 형식 (반드시 이 형식을 따르세요) =====

📋 보고서 제목: [정확히 추출한 제목 없으면, 문맥상 흐름을 분석하여 제목을 정의]

📌 목적 : [결과물 내용을 분석하여 목적을 정의]
📌 요구사항 : [결과물 내용을 분석하여 요구사항을 정의]
📌 피드백 : [피드백 내용을 분석하여 피드백을 정의]

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