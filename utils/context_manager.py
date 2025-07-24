import os
import json
import traceback
from typing import Any
from contextvars import ContextVar
from dotenv import load_dotenv
import openai
import logging

# ============================================================================
# 초기화 및 설정
# ============================================================================

# 환경변수 로드 및 OpenAI 설정
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

# ContextVar 기반 crew 실행 컨텍스트 관리
crew_type_var: ContextVar[str] = ContextVar("crew_type", default="unknown")
todo_id_var: ContextVar[str] = ContextVar("todo_id", default=None)
proc_id_var: ContextVar[str] = ContextVar("proc_inst_id", default=None)
form_id_var: ContextVar[str] = ContextVar("form_id", default=None)

def _handle_error(operation: str, error: Exception) -> None:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    print(error_msg)
    print(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ============================================================================
# 컨텍스트 관리
# ============================================================================

def set_crew_context(crew_type: str, todo_id: str = None, proc_inst_id: str = None, form_id: str = None):
    """ContextVar에 crew 정보 설정 및 토큰 반환"""
    try:
        token_ct = crew_type_var.set(crew_type)
        token_td = todo_id_var.set(todo_id)
        token_pid = proc_id_var.set(proc_inst_id)
        token_fid = form_id_var.set(form_id)
        return token_ct, token_td, token_pid, token_fid
    except Exception as e:
        _handle_error("컨텍스트설정", e)

def reset_crew_context(token_ct, token_td, token_pid, token_fid):
    """ContextVar 설정을 이전 상태로 복원"""
    try:
        crew_type_var.reset(token_ct)
        todo_id_var.reset(token_td)
        proc_id_var.reset(token_pid)
        form_id_var.reset(token_fid)
    except Exception as e:
        _handle_error("컨텍스트리셋", e)

# ============================================================================
# 요약 처리
# ============================================================================

def summarize(outputs: Any, feedbacks: Any) -> str:
    """주어진 outputs와 feedbacks를 LLM으로 요약"""
    try:
        print("\n\n요약을 위한 LLM호출 시작\n\n")
        
        # 데이터 준비
        outputs_str = _convert_to_string(outputs)
        feedbacks_str = _convert_to_string(feedbacks)
        
        # 프롬프트 생성 및 LLM 호출
        prompt = _create_summary_prompt(outputs_str, feedbacks_str)
        summary = _call_openai_api(prompt)
        
        print(f"✅ Context 요약 완료: {len(summary)}자", flush=True)
        return summary
        
    except Exception as e:
        _handle_error("요약처리", e)

def _convert_to_string(data: Any) -> str:
    """데이터를 문자열로 변환"""
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)

def _create_summary_prompt(outputs_str: str, feedbacks_str: str) -> str:
    """요약용 프롬프트 생성"""
    return f"""새 산출물과 피드백을 병합하여, 아래 형식에 맞는 하나의 통합 요약을 생성하세요.

**핵심 원칙:**
1. **보고서 형식**: 결과물 내용이, 마크다운 형식의 보고서 내용 -> 목차(섹션, TOC) 추출 후 목차(섹션)별 요약 진행
2. **단순 텍스트**: 그저 일반 요구사항과 같은 텍스트이면서, 목차 없으면 → 목적+요구사항 추출
3. **피드백**: 피드백이 있으면 → 피드백 내용 포함
4. **목차(섹션, TOC) 원본 유지**: 섹션명을 있는 그대로 추출, 왜곡 금지 (가장 중요!)
5. **피드백 구분**: 특정 에이전트 지정(@@에이전트명) vs 전역적 피드백 구분
6. **분량 제한**: 전체 2000자 이내
7. **수치 우선**: 숫자, 데이터, 구체적 사실 반드시 포함

📌 피드백 분류 기준:
- 특정 에이전트 지정: "@@에이전트명은 이렇게 해라" → 해당 에이전트명 명시
- 전역적 피드백: "이렇게 수정해라", "더 자세히 써라" → 전역적으로 분류

⚠️ 필수: 
1. 목차명(섹션)는 원본 그대로, 수치 데이터 우선, 2000자 이내 (목차 명 왜곡 및 수정 금지)
2. 없는 목차(섹션)를 피드백을 보고 생성하지말고, 있는 그대로 목차(섹션)를 추출하고, 피드백은 같이 전달되니까 현재 단계에서 반영하지 않아도 됩니다.
3. 결과물 내용에 명확히 목차(섹션)가 식별된 구조화된 보고서 형식의 내용일 경우에만 목차(섹션) 추출 후 목차별 요약 진행
4. 목차(섹션)가 없으면 그냥 "없음"으로 처리
5. 결과물은 반드시 그저 요약 및 핵심 정보 추출일 뿐입니다. 피드백을 반영하는게 아닙니다.
6. 전달된 값은 사전 형태로, 중첩된 구조입니다. 오로직 값을 기준으로 목차를 의미적으로 파악해서 추출하세요. 단순히 키를 목차로 두면 안됩니다.(보통 키는 영문자로 되어있음)

**결과물 내용:** {outputs_str}
**피드백 내용:** {feedbacks_str}

===== 요약 형식 (반드시 이 형식을 따르세요) =====

📌 목적 : [결과물 내용을 분석하여 목적을 정의]
📌 요구사항 : [결과물 내용을 분석하여 요구사항을 정의]
📌 피드백 : [피드백 내용을 분석하여 피드백을 정의]

📋 보고서 제목: [정확히 추출한 제목 없으면, 문맥상 흐름을 분석하여 제목을 정의]

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

[계속해서 모든 목차에 대해 동일한 형식으로...]"""

def _get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    return """당신은 전문적인 요약 전문가입니다.

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
5. 실용성: 후속 작업에 활용하기 쉬운 형태로 가공"""

def _call_openai_api(prompt: str) -> str:
    """OpenAI API 호출"""
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": _get_system_prompt()},
            {"role": "user", "content": prompt}
        ],
        max_tokens=3000,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()