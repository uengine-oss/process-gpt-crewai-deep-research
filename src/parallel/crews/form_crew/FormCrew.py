import logging
import traceback
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.parallel.context_manager import set_crew_context, reset_crew_context

# ============================================================================
# 설정 및 초기화
# ============================================================================

# 로거 설정
logger = logging.getLogger("form_crew")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

def _handle_error(operation: str, error: Exception) -> None:
    """통합 에러 처리"""
    error_msg = f"❌ [{operation}] 오류 발생: {str(error)}"
    logger.error(error_msg)
    logger.error(f"상세 정보: {traceback.format_exc()}")
    raise Exception(f"{operation} 실패: {error}")

# ============================================================================
# FormCrew 클래스
# ============================================================================

@CrewBase
class FormCrew:
    """
    JSON 형식의 폼 필드 값을 생성하는 크루입니다.
    사용자 입력과 필드 이름을 기반으로 현실적인 폼 필드 값을 생성합니다.
    """
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def field_value_generator(self) -> Agent:
        """특정 폼 필드에 대한 컨텍스트 기반 값을 생성하는 에이전트"""
        return Agent(
            config=self.agents_config['field_value_generator'],
            verbose=True,
            cache=True
        )

    @task
    def generate_field_value(self) -> Task:
        """여러 폼 필드에 대한 컨텍스트 기반 값을 생성하는 태스크"""
        return Task(
            config=self.tasks_config['generate_field_value'],
            agent=self.field_value_generator()
        )

    @crew
    def crew(self) -> Crew:
        """개별 폼 필드 값을 생성하는 크루를 구성하며, kickoff_async를 WrappedCrew로 오버라이드합니다."""
        # 1) 기본 Agent 및 Task 생성
        agent = self.field_value_generator()
        task  = self.generate_field_value()

        # 2) WrappedCrew 서브클래스 정의: kickoff_async에 ContextVar 관리 및 로깅 추가
        return WrappedCrew(
            agents=[agent],
            tasks=[task],
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
            _handle_error("FormCrew 실행", e)
            
        finally:
            # 컨텍스트 정리
            self._cleanup_context(tokens)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _setup_context(self, inputs):
        """컨텍스트 변수 설정"""
        return set_crew_context(
            crew_type="text",
            todo_id=inputs.get('todo_id') if inputs else None,
            proc_inst_id=inputs.get('proc_inst_id') if inputs else None,
            form_id=inputs.get('form_id') if inputs else None
        )

    def _log_start(self, inputs):
        """시작 로그"""
        if inputs:
            topic = inputs.get('topic', '')
            field_count = len(inputs.get('field_info', []))
            user_count = len(inputs.get('user_info', []))
            logger.info(f"🚀 FormCrew 시작: topic={topic}, fields={field_count}, users={user_count}")
        else:
            logger.info("🚀 FormCrew 시작: 입력 없음")

    def _log_completion(self, inputs):
        """완료 로그"""
        input_keys = list(inputs.keys()) if inputs else None
        logger.info(f"✅ FormCrew 완료: inputs={input_keys}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)
