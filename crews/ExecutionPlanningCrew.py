import logging
import traceback
from functools import wraps
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from utils.context_manager import set_crew_context, reset_crew_context

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
# ExecutionPlanningCrew 클래스
# ============================================================================

@CrewBase
class ExecutionPlanningCrew:
    """
    멀티 포맷 콘텐츠 생성을 위한 종합 실행 계획을 수립하는 전문 크루입니다.
    폼 조합을 분석하고 종속성 및 병렬 처리 전략을 포함한 지능형 실행 계획을 생성합니다.
    """
    agents_config = "agent/planning_agents.yaml"
    tasks_config  = "task/planning_tasks.yaml"

    @agent
    def dependency_analyzer(self) -> Agent:
        """폼 종속성을 분석하고 실행 계획을 수립하는 AI 에이전트입니다."""
        return Agent(
            config=self.agents_config['dependency_analyzer'],
            verbose=True,
            cache=True
        )

    @task
    def create_execution_plan(self) -> Task:
        """모든 폼 유형에 대한 종합 실행 계획을 작성하는 태스크입니다."""
        return Task(
            config=self.tasks_config['create_execution_plan'],
            agent=self.dependency_analyzer()
        )

    @crew
    def crew(self) -> Crew:
        """실행 계획 크루를 생성하고, ContextVar 로깅을 적용한 WrappedCrew 타입을 반환합니다."""
        return WrappedCrew(
            agents=[self.dependency_analyzer()],
            tasks=[self.create_execution_plan()],
            process=Process.sequential,
            verbose=True,
            cache=True
        )

# ============================================================================
# WrappedCrew 클래스
# ============================================================================

class WrappedCrew(Crew):
    """컨텍스트 관리와 로깅이 추가된 크루"""

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
            self._log_completion(inputs)
            return result
            
        except Exception as e:
            _handle_error("ExecutionPlanningCrew 실행", e)
            
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
        input_keys = list(inputs.keys()) if inputs else None
        logger.info(f"🚀 ExecutionPlanningCrew 시작: inputs={input_keys}")

    def _log_completion(self, inputs):
        """완료 로그"""
        input_keys = list(inputs.keys()) if inputs else None
        logger.info(f"✅ ExecutionPlanningCrew 완료: inputs={input_keys}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)
