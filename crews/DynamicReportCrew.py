import logging
import traceback
from typing import Dict, Any, Optional
from crewai import Agent, Crew, Process, Task
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
    
    def __init__(self, section_data: Dict[str, Any], topic: str, previous_context: Optional[Dict[str, Any]] = None):
        """초기화 및 설정"""
        self.previous_context = previous_context
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
            crew._previous_context = self.previous_context
            
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
        """이전 컨텍스트 정보 구성"""
        if not self.previous_context:
            return ""
        
        context_str = str(self.previous_context)
        return f"\n\n[이전 작업 컨텍스트]\n{context_str}"

    def _build_task_description(self, base_description: str, context_info: str, user_id: str, tenant_id: str) -> str:
        """작업 설명 구성"""
        return base_description + context_info + f"""

        **📋 작업 원칙:**
        1. **피드백 절대 반영**: 이전 컨텍스트의 피드백이 특정 에이전트 대상인지 전역적인지 구분하여 반드시 적용
        2. **흐름 연속성**: 이전 작업의 목적과 요구사항을 현재 섹션에 일관되게 유지
        3. **섹션 전문성**: 현재 TOC 섹션 '{self.section_title}'에 최적화된 내용 작성
        4. **이전 결과 활용**: 이전 단계에서 생성된 결과물과 자연스럽게 연결되는 내용 구성

        **🔍 도구 사용 지침:**
        - **mem0 필수 조회**: mem0(query="현재 작성할 섹션과 관련된 구체적 정보")로 시작 - 작성할 내용에 필요한 정보를 동적으로 검색
        - **perplexity 보완**: 필요시 perplexity 도구로 최신 정보 보완
        - **memento 내부 문서 검색**: memento(query="OO 내부 문서를 참고")로 사내 문서를 검색하여 추가 정보 보강
        - **query 명확성**: 구체적이고 명확한 검색어 사용 ⚠️ CRITICAL: null, 빈값, 공백, "null", "None" 등 절대 금지!
          * ✅ 올바른 예시: "AI 기술 동향", "데이터베이스 최적화 방법", "클라우드 보안 전략"
          * ❌ 잘못된 예시: null, "", " ", "null", "None", undefined
        - **URL 접속 금지**: 웹사이트 직접 접속이나 임의 주소 생성 금지
        - **출처 표기**: 출처 표기 필수 (어떤 정보로 부터 참고했는지 출처를 명시, 어떤 문서로 부터 참고했는지 출처를 명시)

        **📊 내용 구성 원칙:**
        - 도구의 사용 결과가 없어도, 이전 컨텍스트와 피드백을 우선적으로 반영하여 창의적으로 내용을 작성
        - 이전 컨텍스트와 피드백을 우선적으로 반영하여 내용 작성
        - 도구 정보 결과에만 의존하지 말고 전문가적 관점에서 창의적 작성
        - 현재 섹션의 목적에 맞는 심층적이고 실무적인 내용 제공
        - 업계 표준과 모범 사례를 활용한 완성도 높은 결과물 작성
        """

    def _build_expected_output(self, expected_output: str) -> str:
        """기대 출력 구성"""
        return expected_output + f"""

        **📊 섹션별 품질 기준:**
        - **이전 컨텍스트 반영**: 피드백과 이전 결과를 섹션 '{self.section_title}'에 자연스럽게 통합
        - **분량**: 최소 3,000-4,000단어 이상의 상세하고 전문적인 내용
        - **심층성**: 표면적 설명이 아닌 해당 분야 전문가 수준의 심층 분석
        - **실무성**: 바로 활용 가능한 구체적 사례와 예시 다수 포함
        - **포괄성**: 관련 법규, 절차, 모범 사례, 주의사항 종합적 다룸

        **📝 출력 형식:**
        - 순수한 마크다운 텍스트 (코드 블록 감싸기 금지)
        - 체계적인 제목 구조와 하위 섹션 구분
        - 마크다운 형식 활용: ## 제목, ### 소제목, **강조**, - 리스트

        **⚠️ 필수 사항:** 도구 검색 결과가 부족해도 반드시 완성된 보고서 제공"""

# ============================================================================
# WrappedCrew 클래스
# ============================================================================

class WrappedCrew(Crew):
    """컨텍스트 관리와 로깅이 추가된 크루"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._section_title = None
        self._previous_context = None

    async def kickoff_async(self, inputs=None):
        """비동기 실행 with 컨텍스트 관리 및 로깅"""
        # 컨텍스트 설정
        tokens = self._setup_context(inputs)
        
        try:
            # 시작 로그
            self._log_start(inputs)
            
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
            form_id=inputs.get('report_form_id') if inputs else None
        )

    def _log_start(self, inputs):
        """시작 로그"""
        logger.info(f"🚀 DynamicReportCrew 시작: section={self._section_title}")
        if self._previous_context:
            context_snippet = str(self._previous_context)[:100]
            logger.info(f"📄 이전 컨텍스트: {context_snippet}...")
        else:
            logger.info("📄 이전 컨텍스트: 없음")

    def _log_completion(self):
        """완료 로그"""
        logger.info(f"✅ DynamicReportCrew 완료: section={self._section_title}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)

