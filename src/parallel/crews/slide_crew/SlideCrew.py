import logging
import traceback
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.parallel.context_manager import set_crew_context, reset_crew_context

# ============================================================================
# 설정 및 초기화
# ============================================================================

# 로거 설정
logger = logging.getLogger("slide_crew")
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
# SlideCrew 클래스
# ============================================================================

@CrewBase
class SlideCrew:
    """
    리포트 내용을 reveal.js 마크다운 형식 슬라이드로 변환하는 크루

    이 크루는 마크다운 리포트를 분석하여 reveal.js 형식에 적합한
    프레젠테이션 슬라이드로 변환합니다.
    """
    agents_config = "config/agents.yaml"
    tasks_config  = "config/tasks.yaml"

    @agent
    def slide_generator(self) -> Agent:
        """리포트 분석과 reveal.js 슬라이드 생성을 담당하는 에이전트"""
        return Agent(
            config=self.agents_config['slide_generator'],
            verbose=True,
            cache=True
        )

    @task
    def generate_reveal_slides(self) -> Task:
        """리포트 분석부터 reveal.js 슬라이드 생성까지 통합 수행하는 태스크"""
        return Task(
            config=self.tasks_config['generate_reveal_slides'],
            agent=self.slide_generator()
        )

    @crew
    def crew(self) -> Crew:
        """슬라이드 생성 크루를 구성"""
        return WrappedCrew(
            agents=[self.slide_generator()],
            tasks=[self.generate_reveal_slides()],
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
            _handle_error("SlideCrew 실행", e)
            
        finally:
            # 컨텍스트 정리
            self._cleanup_context(tokens)

    # ============================================================================
    # 헬퍼 메서드들
    # ============================================================================

    def _setup_context(self, inputs):
        """컨텍스트 변수 설정"""
        return set_crew_context(
            crew_type="slide",
            todo_id=inputs.get('todo_id') if inputs else None,
            proc_inst_id=inputs.get('proc_inst_id') if inputs else None,
            form_id=inputs.get('slide_form_id') if inputs else None
        )

    def _log_start(self, inputs):
        """시작 로그"""
        if inputs and 'report_content' in inputs:
            content_length = len(inputs.get('report_content', '') or "")
            user_count = len(inputs.get('user_info', []))
            logger.info(f"🚀 SlideCrew 시작: content_length={content_length}, users={user_count}")
        else:
            logger.info("🚀 SlideCrew 시작: 입력 없음")

    def _log_completion(self, inputs):
        """완료 로그"""
        input_keys = list(inputs.keys()) if inputs else None
        logger.info(f"✅ SlideCrew 완료: inputs={input_keys}")

    def _cleanup_context(self, tokens):
        """컨텍스트 정리"""
        token_ct, token_td, token_pid, token_fid = tokens
        reset_crew_context(token_ct, token_td, token_pid, token_fid)
