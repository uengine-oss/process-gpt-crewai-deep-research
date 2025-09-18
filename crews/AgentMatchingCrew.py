import logging
import traceback
from typing import Dict, Any
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from utils.context_manager import set_crew_context, reset_crew_context
from llm_factory import create_llm

# ============================================================================
# 설정 및 초기화
# ============================================================================

# 로거 설정
logger = logging.getLogger(__name__)

def _handle_error(operation: str, error: Exception) -> None:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    logger.error(error_msg)
    logger.error(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ============================================================================
# AgentMatchingCrew 클래스
# ============================================================================

@CrewBase
class AgentMatchingCrew:
    """
    이전 컨텍스트 분석과 현재 액티비티 기반 TOC 생성 및 에이전트 매칭을 담당하는 크루

    1. 이전 단계들의 작업 흐름과 컨텍스트를 심층 분석
    2. 현재 액티비티에 최적화된 보고서 목차(TOC) 생성
    3. 각 섹션별 최적 에이전트 매칭 + 맞춤형 Task 할당
    """
    agents_config = "agent/matching_agents.yaml"
    tasks_config  = "task/matching_tasks.yaml"

    def __init__(self):
        super().__init__()

    @agent
    def toc_generator_and_agent_matcher(self) -> Agent:
        """보고서 TOC 생성 및 에이전트 매칭을 담당하는 전문가"""
        # 기본 모델: gpt-4.1
        llm = create_llm(model="gpt-4.1", temperature=0.1)
        agent = Agent(
            config=self.agents_config['toc_generator_and_agent_matcher'],
            verbose=True,
            cache=True,
            llm=llm
        )
        return agent

    @task
    def design_activity_tasks(self) -> Task:
        """컨텍스트 분석과 액티비티별 작업 설계 + 에이전트 매칭을 통합하여 수행"""
        return Task(
            config=self.tasks_config['design_activity_tasks'],
            # Agent 설정은 config 내 또는 별도 매핑으로 처리됩니다.
        )

    @crew
    def crew(self) -> Crew:
        """Agent Matching Crew 구성"""
        return WrappedCrew(
            agents=[self.toc_generator_and_agent_matcher()],
            tasks=[self.design_activity_tasks()],
            process=Process.sequential,
            verbose=True,
            cache=True
        )

# ============================================================================
# WrappedCrew 클래스
# ============================================================================

class WrappedCrew(Crew):
    """컨텍스트 관리와 로깅이 추가된 크루"""

    async def kickoff_async(self, inputs: Dict[str, Any] = None):
        """비동기 실행 with 컨텍스트 관리 및 로깅"""
        # 컨텍스트 설정
        tokens = self._setup_context(inputs)
        
        try:
            # 시작 로그
            self._log_start(inputs)
            
            # 실제 크루 실행
            result = await super().kickoff_async(inputs=inputs)
            
            # 완료 로그
            self._log_completion(inputs)
            return result
            
        except Exception as e:
            _handle_error("AgentMatchingCrew 실행", e)
            
        finally:
            # 컨텍스트 정리
            self._cleanup_context(tokens)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _setup_context(self, inputs):
        """컨텍스트 변수 설정"""
        return set_crew_context(
            crew_type="planning",
            todo_id=inputs.get('todo_id') if inputs else None,
            proc_inst_id=inputs.get('proc_inst_id') if inputs else None
        )

    def _log_start(self, inputs):
        """시작 로그"""
        if inputs and 'previous_context' in inputs and inputs['previous_context']:
            context_snippet = str(inputs['previous_context'])[:100]
            logger.info(f"🚀 AgentMatchingCrew 시작: context_preview={context_snippet}...")
        else:
            logger.info("🚀 AgentMatchingCrew 시작: 이전 컨텍스트 없음")

    def _log_completion(self, inputs):
        """완료 로그"""
        input_keys = list(inputs.keys()) if inputs else None
        logger.info(f"✅ AgentMatchingCrew 완료: inputs={input_keys}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)
