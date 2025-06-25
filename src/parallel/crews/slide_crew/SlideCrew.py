from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from src.parallel.context_manager import set_crew_context, reset_crew_context

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
        """슬라이드 생성 크루를 구성하며, kickoff_async를 WrappedCrew로 오버라이드합니다."""
        # 1) 기본 Agent 및 Task 생성
        agent      = self.slide_generator()
        slide_task = self.generate_reveal_slides()

        # 2) WrappedCrew 서브클래스 정의
        class WrappedCrew(Crew):
            async def kickoff_async(self, inputs=None):
                # ContextVar 설정
                token_ct, token_td, token_pid = set_crew_context(
                    crew_type="slide",
                    todo_id=inputs.get('todo_id') if inputs else None,
                    proc_inst_id=inputs.get('proc_inst_id') if inputs else None
                )
                # 시작 로그
                if inputs and 'report_content' in inputs:
                    length = len(inputs.get('report_content', '') or "")
                    user   = inputs.get('user_info', {}).get('name', '')
                    print(f"[SlideCrew] 시작 - length={length}, user={user}", flush=True)
                else:
                    print("[SlideCrew] 시작 - no inputs", flush=True)
                try:
                    # 실제 부모 클래스 kickoff_async 호출
                    return await super(WrappedCrew, self).kickoff_async(inputs=inputs)
                finally:
                    # 종료 로그 및 ContextVar 복원
                    print(f"[SlideCrew] 종료 - inputs={list(inputs.keys()) if inputs else None}", flush=True)
                    reset_crew_context(token_ct, token_td, token_pid)

        # 3) WrappedCrew 인스턴스 반환
        return WrappedCrew(
            agents=[agent],
            tasks=[slide_task],
            process=Process.sequential,
            verbose=True,
            cache=True
        )
