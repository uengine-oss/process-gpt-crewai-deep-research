from functools import wraps
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from typing import List, Dict, Any
from ...agents_repository import AgentsRepository
from ...context_manager import set_crew_context, reset_crew_context

@CrewBase
class AgentMatchingCrew:
    """
    이전 컨텍스트 분석과 현재 액티비티 기반 TOC 생성 및 에이전트 매칭을 담당하는 크루

    1. 이전 단계들의 작업 흐름과 컨텍스트를 심층 분석
    2. 현재 액티비티에 최적화된 보고서 목차(TOC) 생성
    3. 각 섹션별 최적 에이전트 매칭 + 맞춤형 Task 할당
    """
    agents_config = "agent_matching_config/agents.yaml"
    tasks_config  = "agent_matching_config/tasks.yaml"

    def __init__(self):
        super().__init__()
        self.agents_repository = AgentsRepository()

    @agent
    def toc_generator_and_agent_matcher(self) -> Agent:
        """보고서 TOC 생성 및 에이전트 매칭을 담당하는 전문가"""
        return Agent(
            config=self.agents_config['toc_generator_and_agent_matcher'],
            verbose=True,
            cache=True
        )

    @task
    def design_activity_tasks(self) -> Task:
        """컨텍스트 분석과 액티비티별 작업 설계 + 에이전트 매칭을 통합하여 수행"""
        return Task(
            config=self.tasks_config['design_activity_tasks'],
            # Agent 설정은 config 내 또는 별도 매핑으로 처리됩니다.
        )

    @crew
    def crew(self) -> Crew:
        """Agent Matching Crew 구성: WrappedCrew로 kickoff_async override 적용"""
        class WrappedCrew(Crew):
            async def kickoff_async(self, inputs: Dict[str, Any] = None):
                # 1) ContextVar 설정
                token_ct, token_td, token_pid = set_crew_context(
                    crew_type="planning",
                    todo_id=inputs.get('todo_id') if inputs else None,
                    proc_inst_id=inputs.get('proc_inst_id') if inputs else None
                )
                # 2) 시작 로그: 이전 컨텍스트 일부 출력
                if inputs and 'previous_context' in inputs and inputs['previous_context']:
                    snippet = str(inputs['previous_context'])[:100]
                    print(f"[AgentMatchingCrew] 시작합니다 - prev_context={snippet}", flush=True)
                else:
                    print("[AgentMatchingCrew] 시작합니다", flush=True)
                try:
                    # 3) 실제 Crew.kickoff_async 실행
                    return await super(WrappedCrew, self).kickoff_async(inputs=inputs)
                finally:
                    # 4) 종료 로그 + ContextVar 복원
                    print(f"[AgentMatchingCrew] 종료합니다 - inputs={inputs}", flush=True)
                    reset_crew_context(token_ct, token_td, token_pid)

        # WrappedCrew 인스턴스 반환
        return WrappedCrew(
            agents=[ self.toc_generator_and_agent_matcher() ],
            tasks=[ self.design_activity_tasks() ],
            process=Process.sequential,
            verbose=True,
            cache=True
        )
