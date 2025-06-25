from functools import wraps
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from ...context_manager import set_crew_context, reset_crew_context

@CrewBase
class ExecutionPlanningCrew:
    """
    멀티 포맷 콘텐츠 생성을 위한 종합 실행 계획을 수립하는 전문 크루입니다.
    폼 조합을 분석하고 종속성 및 병렬 처리 전략을 포함한 지능형 실행 계획을 생성합니다.
    """
    agents_config = "execution_planning_config/agents.yaml"
    tasks_config  = "execution_planning_config/tasks.yaml"

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
        # WrappedCrew 클래스를 내부 정의하여 kickoff_async를 오버라이드합니다.
        class WrappedCrew(Crew):
            async def kickoff_async(self, inputs=None):
                # 1) 컨텍스트 설정
                token_ct, token_td, token_pid = set_crew_context(
                    crew_type="planning",
                    todo_id=inputs.get('todo_id') if inputs else None,
                    proc_inst_id=inputs.get('proc_inst_id') if inputs else None
                )
                print(f"[ExecutionPlanningCrew] 시작 - inputs={list(inputs.keys()) if inputs else None}", flush=True)
                
                try:
                    # 2) 실제 실행
                    return await super(WrappedCrew, self).kickoff_async(inputs=inputs)
                finally:
                    # 3) 종료 로그 + 컨텍스트 복원
                    print(f"[ExecutionPlanningCrew] 종료 - inputs={list(inputs.keys()) if inputs else None}", flush=True)
                    reset_crew_context(token_ct, token_td, token_pid)

        # WrappedCrew 인스턴스 생성
        return WrappedCrew(
            agents=[ self.dependency_analyzer() ],
            tasks=[ self.create_execution_plan() ],
            process=Process.sequential,
            verbose=True,
            cache=True
        )
