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

def summarize(outputs: Any, feedbacks: Any, drafts: Any) -> str:
    """주어진 outputs, feedbacks, drafts를 LLM으로 요약"""
    try:
        print("\n\n요약을 위한 LLM호출 시작\n\n")
        
        # 데이터 준비
        outputs_str = _convert_to_string(outputs)
        feedbacks_str = _convert_to_string(feedbacks)
        drafts_str = _convert_to_string(drafts)
        
        # 프롬프트 생성 및 LLM 호출
        prompt = _create_summary_prompt(outputs_str, feedbacks_str, drafts_str)
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

def _create_summary_prompt(outputs_str: str, feedbacks_str: str, drafts_str: str) -> str:
    """요약용 프롬프트 생성"""
    return f"""이전 작업 결과물, 중간결과물, 피드백을 분석하여 핵심 정보만 간결하게 요약하세요.

**분석 원칙:**
1. **구체적 정보 보존**: 수치, 목차명, 섹션명은 원본 그대로 정확히 기록
2. **핵심 내용만 추출**: 불필요한 해석이나 추가 지시사항 없이 사실만 정리
3. **가독성 우선**: 명확하고 간결한 형태로 구조화

**입력 데이터:**
**결과물 내용:** {outputs_str}
**중간결과물 내용:** {drafts_str}
**피드백 내용:** {feedbacks_str}

**요약 형식:**

## 📌 기본 정보
- **목적**: [결과물을 보고 이전 작업의 목적이 무엇이었는지]
- **요구사항**: [결과물을 보고 어떤 요구사항을 충족하려 했는지]

## 📋 완료된 결과물
- **주요 내용**: [결과물의 핵심 내용을 구체적으로]
- **목차/구조**: [목차나 섹션이 있다면 원본 그대로 나열]
  - [목차1]: [해당 섹션 핵심 내용]
  - [목차2]: [해당 섹션 핵심 내용]
  - [계속...]
- **주요 수치/데이터**: [중요한 숫자, 통계, 데이터가 있다면]

## 🔄 중간결과 및 피드백 종합 분석
- **철회된 내용**: [중간결과물에서 문제가 된 부분]
- **문제 원인**: [왜 철회되었는지, 피드백에서 지적된 문제점]
- **구체적 개선방향**: [피드백을 바탕으로 어떻게 개선해야 하는지]
- **추가 요청사항**: [새로 요청된 기능이나 내용이 있다면]

⚠️ **중요**: 모든 목차명, 수치, 데이터는 원본과 정확히 일치하게 기록하고, 해석이나 추가 제안 없이 사실만 정리하세요."""

def _get_system_prompt() -> str:
    """시스템 프롬프트 반환"""
    return """당신은 이전 작업 결과를 정확하게 요약하는 전문가입니다.

주요 역할:
- 완료된 결과물에서 핵심 정보를 정확히 추출
- 중간결과물과 피드백을 종합 분석하여 개선 방향 파악
- 목차, 수치, 데이터 등 구체적 정보를 원본 그대로 보존
- 간결하고 가독성 높은 형태로 핵심만 정리

작업 원칙:
1. **정확성**: 원본 정보를 왜곡 없이 그대로 기록
2. **간결성**: 불필요한 해석이나 추가 제안 없이 사실만 정리
3. **구조화**: 목적, 결과물, 개선방향으로 체계적 정리
4. **실용성**: 다음 작업에 필요한 핵심 컨텍스트만 제공
5. **종합성**: 중간결과와 피드백을 연관지어 통합 분석"""

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