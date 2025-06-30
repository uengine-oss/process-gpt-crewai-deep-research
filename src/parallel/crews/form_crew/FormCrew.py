from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.parallel.context_manager import set_crew_context, reset_crew_context

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
        class WrappedCrew(Crew):
            async def kickoff_async(self, inputs=None):
                # ContextVar 설정
                token_ct, token_td, token_pid = set_crew_context(
                    crew_type="text",
                    todo_id=inputs.get('todo_id') if inputs else None,
                    proc_inst_id=inputs.get('proc_inst_id') if inputs else None
                )
                # 시작 로그
                if inputs:
                    topic = inputs.get('topic', '')
                    count = len(inputs.get('field_info', []))
                    user_info = inputs.get('user_info', [])
                    print(f"[FormCrew] 시작합니다 - topic={topic}, fields={count}, user_info={user_info}", flush=True)
                else:
                    print("[FormCrew] 시작합니다 - no inputs", flush=True)
                try:
                    # 실제 부모 클래스 kickoff_async 호출
                    return await super(WrappedCrew, self).kickoff_async(inputs=inputs)
                finally:
                    # 종료 로그 및 ContextVar 복원
                    print(f"[FormCrew] 종료 - inputs={list(inputs.keys()) if inputs else None}", flush=True)
                    reset_crew_context(token_ct, token_td, token_pid)

        # 3) WrappedCrew 인스턴스 반환
        return WrappedCrew(
            agents=[agent],
            tasks=[task],
            process=Process.sequential,
            verbose=True,
            cache=True
        )
