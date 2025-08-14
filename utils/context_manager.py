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

def handle_error(operation: str, error: Exception) -> None:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    print(error_msg)
    print(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ContextVar 기반 crew 실행 컨텍스트 관리
crew_type_var: ContextVar[str] = ContextVar("crew_type", default="unknown")
todo_id_var: ContextVar[str] = ContextVar("todo_id", default=None)
proc_id_var: ContextVar[str] = ContextVar("proc_inst_id", default=None)
form_id_var: ContextVar[str] = ContextVar("form_id", default=None)
form_key_var: ContextVar[str] = ContextVar("form_key", default=None)



# ============================================================================
# 컨텍스트 관리
# ============================================================================

def set_crew_context(crew_type: str, todo_id: str = None, proc_inst_id: str = None, form_id: str = None, form_key: str = None):
    """ContextVar에 crew 정보 설정 및 토큰 반환"""
    try:
        token_ct = crew_type_var.set(crew_type)
        token_td = todo_id_var.set(todo_id)
        token_pid = proc_id_var.set(proc_inst_id)
        token_fid = form_id_var.set(form_id)
        # slide/report 에서만 form_key 저장
        if crew_type in ("slide", "report"):
            form_key_var.set(form_key)
        else:
            form_key_var.set(None)
        return token_ct, token_td, token_pid, token_fid
    except Exception as e:
        handle_error("컨텍스트설정", e)

def reset_crew_context(token_ct, token_td, token_pid, token_fid):
    """ContextVar 설정을 이전 상태로 복원"""
    try:
        crew_type_var.reset(token_ct)
        todo_id_var.reset(token_td)
        proc_id_var.reset(token_pid)
        form_id_var.reset(token_fid)
        form_key_var.set(None)
    except Exception as e:
        handle_error("컨텍스트리셋", e)

# ============================================================================
# 요약 처리
# ============================================================================

import asyncio

async def summarize_async(outputs: Any, feedbacks: Any, contents: Any = None) -> tuple[str, str]:
    """LLM으로 컨텍스트 요약 - 병렬 처리로 별도 반환 (비동기)"""
    try:
        logger.info("요약을 위한 LLM 병렬 호출 시작")
        
        # 데이터 준비
        outputs_str = _convert_to_string(outputs)
        feedbacks_str = _convert_to_string(feedbacks) if any(item for item in (feedbacks or []) if item and item != {}) else ""
        contents_str = _convert_to_string(contents) if contents and contents != {} else ""
        
        # 병렬 처리
        output_summary, feedback_summary = await _summarize_parallel(outputs_str, feedbacks_str, contents_str)
        
        logger.info(f"이전결과 요약 완료: {len(output_summary)}자, 피드백 요약 완료: {len(feedback_summary)}자")
        return output_summary, feedback_summary
        
    except Exception as e:
        handle_error("요약처리", e)
        return "", ""

async def _summarize_parallel(outputs_str: str, feedbacks_str: str, contents_str: str = "") -> tuple[str, str]:
    """병렬로 요약 처리 - 별도 반환"""
    tasks = []
    
    # 1. 이전 결과물 요약 태스크 (데이터가 있을 때만)
    if outputs_str and outputs_str.strip():
        output_prompt = _create_output_summary_prompt(outputs_str)
        tasks.append(_call_openai_api_async(output_prompt, "이전 결과물"))
    else:
        tasks.append(_create_empty_task(""))
    
    # 2. 피드백 요약 태스크 (피드백 또는 현재 결과물이 있을 때만)
    if (feedbacks_str and feedbacks_str.strip()) or (contents_str and contents_str.strip()):
        feedback_prompt = _create_feedback_summary_prompt(feedbacks_str, contents_str)
        tasks.append(_call_openai_api_async(feedback_prompt, "피드백"))
    else:
        tasks.append(_create_empty_task(""))
    
    # 3. 두 태스크를 동시에 실행하고 완료될 때까지 대기
    output_summary, feedback_summary = await asyncio.gather(*tasks)
    
    # 4. 별도로 반환
    return output_summary, feedback_summary

async def _create_empty_task(result: str) -> str:
    """빈 태스크 생성 (즉시 완료)"""
    return result

def _convert_to_string(data: Any) -> str:
    """데이터를 문자열로 변환"""
    if isinstance(data, str):
        return data
    return json.dumps(data, ensure_ascii=False)



def _create_output_summary_prompt(outputs_str: str) -> str:
    """이전 결과물 요약 프롬프트 - 목차, 수치 관련 지시사항 보강"""
    return f"""다음 작업 결과를 체계적으로 정리해주세요:

{outputs_str}

**정리 원칙:**
- **구체적 정보 완전 보존**: 수치, 목차명, 섹션명, 인물명, 물건명, 날짜, 시간 등 객관적 정보는 반드시 원본 그대로 정확히 기록
- **정보 손실 방지**: 짧은 내용은 요약하지 말고 그대로 유지 (오히려 정보 손실 위험)
- **의미 보존**: 왜곡이나 의미 변경 절대 금지, 원본 의미 그대로 보존
- **효율적 정리**: 긴 내용만 적절히 요약하여 핵심 정보 전달
- **중복 제거**: 중복된 부분만 정리하고 핵심 내용은 모두 보존
- **통합성**: 하나의 통합된 문맥으로 작성하여 다음 작업자가 즉시 이해 가능

**특별 주의사항 - 목차 및 구조 정보:**
- **목차명**: 원본 목차 제목을 정확히 그대로 기록 (1자도 변경 금지)
- **섹션 구조**: 원본의 섹션 구조와 순서를 그대로 유지
- **번호 체계**: 목차 번호, 항목 번호 등을 원본과 정확히 일치시켜 기록
- **계층 구조**: 대분류, 소분류 등 계층 관계를 명확히 보존

**특별 주의사항 - 수치 및 데이터:**
- **모든 숫자**: 통계, 비율, 개수, 날짜, 시간 등 모든 수치를 정확히 기록
- **단위 포함**: 수치와 함께 단위(%, 개, 명, 원, 시간 등)도 함께 기록
- **범위/기간**: 날짜 범위, 시간대, 기간 등을 정확히 명시
- **비교 데이터**: 전년 대비, 이전 대비 등 비교 수치도 누락 없이 포함

**정리 방식:**
1. **목적과 배경**: 작업의 목적과 배경 정보를 명확히 기록
2. **핵심 결과물**: 완성된 내용의 구조와 세부사항을 체계적으로 정리
   - 목차가 있다면 원본 그대로 나열
   - 각 섹션별 핵심 내용 요약
3. **중요 데이터**: 수치, 통계, 목록 등 객관적 정보는 누락 없이 포함
4. **구조 유지**: 원본의 논리적 흐름과 구조를 최대한 보존

**출력 형식**: 통합된 하나의 완전한 문서로 작성 (불필요한 부연설명 제거, 객관적 사실만 포함)

**길이 제한**: 2000자 이내로 작성 (목차, 수치 등 핵심 정보는 유지하되 불필요한 부분만 압축)

⚠️ **절대 금지사항**: 목차명, 수치, 날짜, 인명 등의 변경이나 생략 절대 금지. 모든 구체적 정보는 원본과 100% 일치해야 함."""

def _create_feedback_summary_prompt(feedbacks_str: str, contents_str: str = "") -> str:
    """피드백 정리 프롬프트 - 기존 프롬프트를 참고하여 보강"""
    
    # 피드백과 현재 결과물 모두 준비
    feedback_section = f"""=== 피드백 내용 ===
{feedbacks_str}""" if feedbacks_str and feedbacks_str.strip() else ""
    
    content_section = f"""=== 현재 결과물/작업 내용 ===
{contents_str}""" if contents_str and contents_str.strip() else ""
    
    return f"""다음은 사용자의 피드백과 결과물입니다. 이를 종합 분석하여 통합된 피드백을 작성해주세요:

{feedback_section}

{content_section}

**상황 분석 및 처리 방식:**
- **현재 결과물 품질 평가**: 어떤 점이 문제인지, 개선이 필요한지 구체적으로 판단
- **피드백 의도 파악**: 피드백의 진짜 의도와 숨은 요구사항을 정확히 파악
- **문제점 진단**: 결과물 자체, 작업 방식, 접근법 중 무엇이 문제인지 판단
- **개선 방향 제시**: 구체적이고 실행 가능한 개선 방안을 명확히 제시
- **현실적 분석**: 현재 결과물에 매몰되지 말고, 실제 어떤 부분이 문제인지 객관적 파악

**피드백 통합 및 분석 원칙:**
- **시간 흐름 이해**: 가장 최신 피드백을 최우선으로 하여 피드백들 간의 연결고리와 문맥을 파악
- **종합적 분석**: 결과물과 피드백을 함께 고려하여 핵심 문제점과 개선사항 도출
- **구체성 확보**: 추상적 지시가 아닌 구체적이고 실행 가능한 개선사항 제시
- **완전성 추구**: 자연스럽고 통합된 하나의 완전한 피드백으로 작성
- **명확성 확보**: 다음 작업자가 즉시 이해하고 실행할 수 있도록 명확하게 작성

**중요한 상황별 처리 방식:**
- **품질 문제**: 결과물 품질에 대한 불만 → 구체적인 품질 개선 방향 제시
- **방식 문제**: 작업 방식에 대한 불만 → 접근법 변경 및 새로운 방법론 제안
- **저장 문제**: 이전에 저장했는데 잘못 저장된 경우 → 정확한 수정 방법 제시
- **기능 추가**: 이전에 조회만 했는데 저장이 필요한 경우 → 필요한 저장 기능 명시
- **부분 수정**: 특정 부분만 수정이 필요한 경우 → 정확한 수정 범위와 방법 제시
- **전면 재작업**: 완전히 다시 시작해야 하는 경우 → 새로운 접근 방향 제시

**출력 형식**: 현재 상황을 종합적으로 분석한 완전한 피드백 문장 (최대 2500자까지 허용하여 상세히 작성)
**목표**: 다음 작업자가 이 피드백만 보고도 즉시 정확한 작업을 수행할 수 있도록 하는 것"""



def _get_output_system_prompt() -> str:
    """결과물 요약용 시스템 프롬프트 - 기존 프롬프트를 참고하여 보강"""
    return """당신은 작업 결과물을 정확하게 정리하는 전문가입니다.

핵심 사명:
- **정보 손실 방지**: 짧은 내용은 요약하지 말고 그대로 유지 (오히려 정보 손실 위험)
- **의미 보존 최우선**: 왜곡이나 의미 변경 절대 금지, 원본 의미 그대로 보존
- **객관적 정보 완전 보존**: 수치, 목차, 인물명, 물건명, 날짜, 시간 등 객관적 정보는 반드시 포함
- **효율적 정리**: 긴 내용만 적절히 요약하여 핵심 정보 전달
- **통합성 확보**: 하나의 통합된 문맥으로 작성하여 다음 작업자가 즉시 이해 가능

작업 원칙:
1. **정확성**: 원본 정보를 왜곡 없이 그대로 기록
2. **완전성**: 중복된 부분만 정리하고 핵심 내용은 모두 보존
3. **구조화**: 원본의 논리적 흐름과 구조를 최대한 보존
4. **실용성**: 다음 작업자가 즉시 이해할 수 있도록 명확하게
5. **객관성**: 객관적 사실만 포함, 불필요한 부연설명만 제거

금지사항:
- 짧은 내용의 무분별한 요약
- 수치, 날짜, 인명 등 객관적 정보 누락
- 원본 의미의 왜곡이나 변경
- 개인적 해석이나 추가 제안"""

def _get_feedback_system_prompt() -> str:
    """피드백 정리용 시스템 프롬프트 - 기존 프롬프트를 참고하여 보강"""
    return """당신은 피드백 분석 및 통합 전문가입니다.

핵심 사명:
- **최신 피드백 최우선**: 시간 흐름을 파악하여 가장 최신 피드백을 최우선으로 반영
- **문맥 파악**: 피드백들 간의 연결고리와 전체적인 문맥을 정확히 이해
- **진짜 의도 파악**: 표면적 피드백이 아닌 진짜 의도와 숨은 요구사항을 정확히 파악
- **종합적 분석**: 결과물과 피드백을 함께 고려하여 핵심 문제점과 개선사항 도출
- **실행 가능성**: 추상적 지시가 아닌 구체적이고 실행 가능한 개선사항 제시

작업 원칙:
1. **시간성**: 최신 피드백을 최우선으로 하여 시간 흐름 파악
2. **통합성**: 자연스럽고 통합된 하나의 완전한 피드백으로 작성
3. **구체성**: 구체적이고 실행 가능한 개선사항을 누락 없이 포함
4. **명확성**: 다음 작업자가 즉시 이해할 수 있도록 명확하게
5. **완전성**: 다음 작업자가 이 피드백만 보고도 즉시 정확한 작업을 수행할 수 있도록

상황별 대응:
- 품질 문제 → 구체적인 품질 개선 방향 제시
- 방식 문제 → 접근법 변경 및 새로운 방법론 제안
- 기능 문제 → 필요한 기능과 구현 방법 명시
- 부분 수정 → 정확한 수정 범위와 방법 제시
- 전면 재작업 → 새로운 접근 방향과 전략 제시

목표: 다음 작업자가 즉시 정확하고 효과적인 작업을 수행할 수 있도록 하는 완벽한 가이드 제공"""



async def _call_openai_api_async(prompt: str, task_name: str) -> str:
    """OpenAI API 병렬 호출"""
    try:
        # OpenAI 클라이언트를 async로 생성
        client = openai.AsyncOpenAI()
        
        # 작업 유형에 따른 시스템 프롬프트 선택
        if task_name == "피드백":
            system_prompt = _get_feedback_system_prompt()
        else:  # "이전 결과물" 등 다른 모든 경우
            system_prompt = _get_output_system_prompt()
        
        response = await client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content.strip()
        logger.info(f"{task_name} 요약 완료: {len(result)}자")
        return result
        
    except Exception as e:
        handle_error(f"{task_name} OpenAI API 호출", e)
        return "요약 생성 실패"