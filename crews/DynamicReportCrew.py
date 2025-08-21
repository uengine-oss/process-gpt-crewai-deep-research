import logging
import traceback
import json
from typing import Dict, Any, Optional
from crewai import Agent, Crew, Process, Task
from pydantic import PrivateAttr
from tools.safe_tool_loader import SafeToolLoader
from utils.context_manager import set_crew_context, reset_crew_context

# ============================================================================
# 설정 및 초기화
# ============================================================================
logger = logging.getLogger(__name__)

def _handle_error(operation: str, error: Exception) -> None:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    logger.error(error_msg)
    logger.error(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ============================================================================
# Agent 커스텀 클래스
# ============================================================================

class AgentWithProfile(Agent):
    """프로필 필드가 추가된 Agent 클래스"""
    profile: Optional[str] = None
    user_id: Optional[str] = None
    name: Optional[str] = None
    tenant_id: Optional[str] = None

# ============================================================================
# DynamicReportCrew 클래스
# ============================================================================

class DynamicReportCrew:
    """AgentMatchingCrew 결과물에서 동적으로 Agent와 Task를 생성하는 크루"""
    
    def __init__(self, section_data: Dict[str, Any], topic: str, previous_outputs: Optional[str] = None, previous_feedback: Optional[str] = None):
        """초기화 및 설정"""
        self.previous_outputs = previous_outputs
        self.previous_feedback = previous_feedback
        self.topic = topic
        self.toc_info = section_data.get("toc", {})
        self.agent_config = section_data.get("agent", {})
        self.task_config = section_data.get("task", {})
        self.section_title = self.toc_info.get("title", "Unknown Section")
        
        # 도구 로더 초기화 (tenant_id, user_id 전달)
        tenant_id = self.agent_config.get('tenant_id', 'localhost')
        user_id = self.agent_config.get('agent_id', '')
        self.safe_tool_loader = SafeToolLoader(tenant_id=tenant_id, user_id=user_id)
        self.tool_names = self.agent_config.get('tool_names', [])
        self.actual_tools = self.safe_tool_loader.create_tools_from_names(self.tool_names)
        

    def create_crew(self) -> Crew:
        """동적으로 Crew 생성"""
        try:
            agent = self.create_dynamic_agent()
            task = self.create_section_task(agent)
            
            # WrappedCrew 생성 (컨텍스트 정보는 클로저로 전달)
            crew = WrappedCrew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=True,
                cache=True
            )
            
            # 컨텍스트 정보를 crew 인스턴스에 설정
            crew._section_title = self.section_title
            crew.previous_outputs = self.previous_outputs
            crew.previous_feedback = self.previous_feedback
            
            return crew
        except Exception as e:
            _handle_error("DynamicReportCrew 생성", e)

    # ============================================================================
    # Agent 및 Task 생성 메서드들
    # ============================================================================

    def create_dynamic_agent(self) -> AgentWithProfile:
        """동적으로 Agent 생성"""
        agent_role = self.agent_config.get("role", "Unknown Role")
        agent_goal = self.agent_config.get("goal", "Unknown Goal")
        agent_backstory = self.agent_config.get("persona", "Unknown Background")
        llm_model = self.agent_config.get("model", "openai/gpt-4.1")

        logger.info(f"👤 Agent 생성: {len(self.actual_tools)}개 도구 할당")
        
        agent = AgentWithProfile(
            role=agent_role,
            goal=agent_goal,
            backstory=agent_backstory,
            llm=llm_model,
            tools=self.actual_tools,
            verbose=True,
            cache=True
        )
        
        # 프로필 설정
        agent.profile = self.agent_config.get('agent_profile', '')
        agent.user_id = self.agent_config.get('agent_id', '')
        agent.name = self.agent_config.get('name', '')
        agent.tenant_id = self.agent_config.get('tenant_id', '')
        
        return agent

    def create_section_task(self, agent: AgentWithProfile) -> Task:
        """동적으로 섹션 작성 Task 생성"""
        base_description = self.task_config.get("description", "")
        expected_output = self.task_config.get("expected_output", "")

        # 이전 컨텍스트 추가
        context_info = self._build_context_info()
        
        # 작업 지침 구성
        safe_description = self._build_task_description(base_description, context_info, agent.user_id, agent.tenant_id)
        enhanced_expected_output = self._build_expected_output(expected_output)
        
        return Task(
            description=safe_description,
            expected_output=enhanced_expected_output,
            agent=agent
        )

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _build_context_info(self) -> str:
        """이전 컨텍스트 정보 구성 - 피드백과 이전 결과물 분리"""
        context_parts = []
        
        if self.previous_outputs:
            context_parts.append(f"[이전 결과물]\n{self.previous_outputs}")
        
        if self.previous_feedback:
            context_parts.append(f"[피드백]\n{self.previous_feedback}")
        
        if not context_parts:
            return ""
        
        return "\n\n" + "\n\n".join(context_parts)

    def _build_task_description(self, base_description: str, context_info: str, user_id: str, tenant_id: str) -> str:
        """작업 설명 구성"""
        return base_description + context_info + f"""

        **📋 작업 원칙:**
        1. **피드백 최우선 반영**: [피드백] 내용을 가장 우선하여 현재 섹션에 적극 반영하고 개선사항 적용
        2. **이전 결과물 연속성**: [이전 결과물]을 분석하여 문맥을 파악하고 자연스럽게 연결되는 내용 구성
        3. **분리된 처리**: 피드백과 이전 결과물을 각각 별도로 분석하여 목적에 맞게 활용
        4. **섹션 전문성**: 현재 TOC 섹션 '{self.section_title}'에 최적화된 내용 작성

        **🔍 도구 사용 지침 (단계별 진행):**
        
        **1단계: 작업 전 피드백 관련 지식 검토**
        - **mem0 피드백 검토**: mem0(query="섹션 '{self.section_title}' 작성 시 주의사항")으로 해당 섹션 작성 관련 주의점 확인
        - **피드백 관련 지식 조회**: mem0(query="피드백 내용과 관련된 지식")으로 피드백과 연관된 기존 지식 검토
        - **검토 결과 없으면**: 자유롭게 전문지식과 배경지식을 활용하여 작업 진행
        
        **2단계: 객관적 정보 적극 수집**
        - **mem0 구체적 정보**: mem0(query="구체적 수치/사물/인물명")으로 객관적 정보 우선 검색
        - **memento 내부 데이터**: memento(query="관련 내부 문서/데이터")로 사내 구체적 정보 확인
        - **정보 부족 시**: 배경지식 및 다른 도구(perplexity 등) 활용하여 보완
        
        **3단계: 최신 정보 보완**
        - **perplexity 최신 동향**: 필요시 perplexity로 최신 정보 및 트렌드 보완
        - **다른 도구 활용**: mem0/memento에서 정보가 부족한 경우 배경지식과 전문지식 적극 활용
        
        **🎯 도구 활용 원칙:**
        - **query 명확성**: 구체적이고 명확한 검색어 사용 ⚠️ CRITICAL: null, 빈값, 공백, "null", "None" 등 절대 금지!
          * ✅ 올바른 예시: "AI 기술 동향 2024", "데이터베이스 최적화 구체적 방법", "클라우드 보안 실제 사례"
          * ❌ 잘못된 예시: null, "", " ", "null", "None", undefined
        - **객관적 정보 우선**: 수치, 사물명, 인물명, 날짜 등 구체적 정보는 mem0/memento에서 우선 검색
        - **URL 접속 금지**: 웹사이트 직접 접속이나 임의 주소 생성 금지
        - **출처 표기**: 출처 표기 필수 (어떤 정보로부터 참고했는지 출처를 명시, 어떤 문서로부터 참고했는지 출처를 명시)

        **📊 내용 구성 원칙:**
        - **피드백 우선**: [피드백] 내용을 가장 우선적으로 반영하여 사용자 요구사항에 맞는 내용 작성
        - **이전 결과물 활용**: [이전 결과물]의 문맥과 흐름을 파악하여 연속성 있는 내용 구성
        - **분리된 분석**: 피드백과 이전 결과물을 별도로 분석하여 각각의 목적에 맞게 활용
        - **단계별 도구 활용**: 1단계(피드백 검토) → 2단계(객관적 정보 수집) → 3단계(최신 정보 보완) 순서로 진행
        - **객관적 정보 우선**: 구체적 수치, 사물명, 인물명 등은 mem0/memento에서 적극 검색 후 활용
        - **전문지식 보완**: 도구 검색 결과가 부족한 경우 배경지식과 전문가적 관점에서 창의적 작성
        - **섹션 최적화**: 현재 섹션의 목적에 맞는 심층적이고 실무적인 내용 제공
        - **품질 보장**: 업계 표준과 모범 사례를 활용한 완성도 높은 결과물 작성
        """

    def _build_expected_output(self, expected_output: str) -> str:
        """기대 출력 구성"""
        return expected_output + f"""

        **📊 섹션별 품질 기준:**
        - **피드백 최우선 통합**: [피드백] 내용을 섹션 '{self.section_title}'에 적극 반영하고 개선사항 적용
        - **이전 결과물 연결**: [이전 결과물]의 문맥을 파악하여 자연스럽게 연결되는 내용 구성
        - **분리된 활용**: 피드백과 이전 결과물을 각각 분석하여 목적에 맞게 활용
        - **분량**: 최소 3,000-4,000단어 이상의 상세하고 전문적인 내용
        - **심층성**: 표면적 설명이 아닌 해당 분야 전문가 수준의 심층 분석
        - **실무성**: 바로 활용 가능한 구체적 사례와 예시 다수 포함
        - **포괄성**: 관련 법규, 절차, 모범 사례, 주의사항 종합적 다룸

        **📝 출력 형식:**
        - 순수한 마크다운 텍스트 (코드 블록 감싸기 금지)
        - 체계적인 제목 구조와 하위 섹션 구분
        - 마크다운 형식 활용: ## 제목, ### 소제목, **강조**, - 리스트

        **🚨 중요한 출력 형식 규칙:**
        - 절대로 코드 블록(```)으로 마크다운을 감싸지 말 것

        **⚠️ 필수 사항:** 
        - 작업 전 반드시 mem0로 피드백 관련 지식 검토 후 진행
        - 객관적 정보는 mem0/memento에서 우선 검색하고 부족한 경우 전문지식 활용
        - 도구 검색 결과가 부족해도 반드시 완성된 보고서 제공"""

# ============================================================================
# WrappedCrew 클래스
# ============================================================================

class WrappedCrew(Crew):
    """컨텍스트 관리와 로깅이 추가된 크루"""

    _section_title: str = PrivateAttr(default=None)
    previous_outputs: Optional[str] = None
    previous_feedback: Optional[str] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def kickoff_async(self, inputs=None):
        """비동기 실행 with 컨텍스트 관리 및 로깅"""
        # 컨텍스트 설정
        tokens = self._setup_context(inputs)
        
        try:
            # 시작 로그
            self._log_start(inputs)
            # 사용자 정보 간단 주입: Task 설명 말미에 지시 한 줄 추가
            if inputs and inputs.get('user_info'):
                try:
                    user_info_text = json.dumps(inputs.get('user_info'), ensure_ascii=False)
                    for task in getattr(self, 'tasks', []) or []:
                        base_desc = getattr(task, 'description', '') or ''
                        addition = f"\n\n[담당자 정보]\n{user_info_text}\n\n지시: 위 담당자 정보를 참고해 어조/문맥/호칭을 적절히 반영하여 작성하세요."
                        setattr(task, 'description', base_desc + addition)
                except Exception:
                    pass
            
            # 실제 크루 실행
            result = await super().kickoff_async(inputs=inputs)
            
            # 완료 로그
            self._log_completion()
            return result
            
        except Exception as e:
            _handle_error("DynamicReportCrew 실행", e)
            
        finally:
            # 컨텍스트 정리
            self._cleanup_context(tokens)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _setup_context(self, inputs):
        """컨텍스트 변수 설정"""
        return set_crew_context(
            crew_type="report",
            todo_id=inputs.get('todo_id') if inputs else None,
            proc_inst_id=inputs.get('proc_inst_id') if inputs else None,
            form_id=inputs.get('report_form_id') if inputs else None,
            form_key=inputs.get('report_form_id') if inputs else None
        )

    def _log_start(self, inputs):
        """시작 로그"""
        logger.info(f"🚀 DynamicReportCrew 시작: section={self._section_title}")
        if hasattr(self, 'previous_outputs') and self.previous_outputs:
            outputs_snippet = str(self.previous_outputs)[:100]
            logger.info(f"📄 이전 결과물: {outputs_snippet}...")
        if hasattr(self, 'previous_feedback') and self.previous_feedback:
            feedback_snippet = str(self.previous_feedback)[:100]
            logger.info(f"💬 피드백: {feedback_snippet}...")
        if not hasattr(self, 'previous_outputs') or not self.previous_outputs:
            logger.info("📄 이전 결과물: 없음")
        if not hasattr(self, 'previous_feedback') or not self.previous_feedback:
            logger.info("💬 피드백: 없음")

    def _log_completion(self):
        """완료 로그"""
        logger.info(f"✅ DynamicReportCrew 완료: section={self._section_title}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)

