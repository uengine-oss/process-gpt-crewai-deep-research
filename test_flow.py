# test_flow.py

import json
import logging

from crewai import Crew
from crewai.memory.external.external_memory import ExternalMemory

# 이벤트 버스/이벤트 타입 임포트
from crewai.utilities.events import CrewAIEventsBus
from crewai.utilities.events import LLMCallStartedEvent, LLMCallCompletedEvent
from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
from crewai.utilities.events.agent_events import AgentExecutionStartedEvent, AgentExecutionCompletedEvent
from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent

# —————————————————————————
# 1) 로그 레벨 설정
# —————————————————————————
logging.basicConfig(level=logging.INFO)


# —————————————————————————
# 2) 글로벌 이벤트 리스너 등록 (프롬프트 & 진행 상황 모두)
# —————————————————————————
bus = CrewAIEventsBus()

# Task/Agent/Tool/LLM 시작·종료 이벤트 모두 잡아서 간단히 상태 로그
for evt in [
    TaskStartedEvent, TaskCompletedEvent,
    AgentExecutionStartedEvent, AgentExecutionCompletedEvent,
    ToolUsageStartedEvent, ToolUsageFinishedEvent,
    LLMCallStartedEvent, LLMCallCompletedEvent
]:
    @bus.on(evt)
    def _generic_handler(source, event, evt=evt):
        et = event.type
        if et == "task_started":
            print(f"📝 태스크 시작: {event.task.id}")
        elif et == "task_completed":
            print("✅ 태스크 완료")
        elif et == "agent_execution_started":
            print(f"🤖 에이전트 시작: {event.agent.role}")
        elif et == "agent_execution_completed":
            print("✅ 에이전트 종료")
        elif et == "tool_usage_started":
            print(f"🔧 도구 시작: {getattr(event, 'tool_name', '')}")
        elif et == "tool_usage_finished":
            print(f"✅ 도구 종료: {getattr(event, 'tool_name', '')}")
        elif et == "llm_call_started":
            # 1) 전체 메시지 구조를 예쁘게 찍어보기
            print("\n🔍 [LLM → Full messages payload]")
            print(json.dumps(event.messages, ensure_ascii=False, indent=2))

            # 2) role/content 별로도 한 줄씩 쭉 찍어보기
            for msg in event.messages:
                role = msg.get("role")
                content = msg.get("content")
                print(f"\n--- {role} message ---\n{content}\n")

            print("🔍" + "―" * 60)
        elif et == "llm_call_completed":
            print("✅ LLM 호출 완료")


# —————————————————————————
# 3) run_flow 정의
# —————————————————————————
def run_flow(proc_inst_id: str, todos: list[str], memory=None):
    # 3-1) 메모리 설정
    if memory is None:
        memory = ExternalMemory(
            embedder_config={
                "provider": "mem0",
                "config": {"user_id": proc_inst_id}
            }
        )

    # 3-2) 각 스텝별 config
    requirement_config = {
        "agents": [{
            "name": "requirement_agent",
            "role": "assistant",
            "goal": "요구사항을 정확히 이해하고 정리합니다.",
            "backstory": "당신은 요구사항 분석 전문가입니다.",
            "type": "llm",
            "model": "gpt-4"
        }],
        "tasks": [{
            "name": "analyze_requirements",
            "agent": "assistant",
            # instruction 에 memory 플레이스홀더 포함
            "instruction": (
                "이전 메모리 내용:\n{memory}\n\n"
                "작업: {task}\n"
                "위 내용을 참고하여, 주어진 요구사항을 체계적으로 분석하고 요약해 주세요."
            ),
            "description": "주어진 요구사항을 분석하고 요약합니다.",
            "expected_output": "text"
        }]
    }

    proposal_config = {
        "agents": [{
            "name": "proposal_agent",
            "role": "assistant",
            "goal": "분석된 요구사항 기반으로 제안서를 작성합니다.",
            "backstory": "당신은 제안서 작성 전문가입니다.",
            "type": "llm",
            "model": "gpt-4"
        }],
        "tasks": [{
            "name": "draft_proposal",
            "agent": "assistant",
            "instruction": (
                "이전 분석 결과:\n{memory}\n\n"
                "작업: {task}\n"
                "위 내용을 참고하여 제안서 초안을 작성하세요."
            ),
            "description": "제안서 초안을 작성합니다.",
            "expected_output": "text"
        }]
    }

    review_config = {
        "agents": [{
            "name": "review_agent",
            "role": "assistant",
            "goal": "작성된 제안서를 검토하고 피드백을 제공합니다.",
            "backstory": "당신은 제안서 리뷰 전문가입니다.",
            "type": "llm",
            "model": "gpt-4"
        }],
        "tasks": [{
            "name": "review_proposal",
            "agent": "assistant",
            "instruction": (
                "이전 제안서 초안:\n{memory}\n\n"
                "작업: {task}\n"
                "위 초안을 검토하고 개선사항을 제안하세요."
            ),
            "description": "제안서를 검토하고 개선사항을 제안합니다.",
            "expected_output": "text"
        }]
    }

    # 3-3) Crew 생성
    crews = [
        Crew(name="RequirementCrew", memory=True, external_memory=memory, config=requirement_config),
        Crew(name="ProposalCrew",    memory=True, external_memory=memory, config=proposal_config),
        Crew(name="ReviewCrew",      memory=True, external_memory=memory, config=review_config),
    ]

    # 3-4) 순차 실행
    for idx, (crew, todo) in enumerate(zip(crews, todos), start=1):
        print(f"\n--- Flow {proc_inst_id} | Step {idx}: {crew.name} ---")
        print(f"할 일: {todo}")
        result = crew.kickoff(inputs={"task": todo})
        print(f"[{crew.name}] 결과:\n{result}\n")


# —————————————————————————
# 4) main
# —————————————————————————
if __name__ == "__main__":
    todos = [
        "요구사항 전달",
        "요구사항 기반 제안서 작성",
        "제안서 피드백 및 검토"
    ]
    proc_inst_id = "proc_002"

    print("=== 기본 mem0 벡터 DB 테스트 (이벤트 로깅 포함) ===")
    run_flow(proc_inst_id, todos)

    print("\n=== 외부 Pinecone 벡터 DB 테스트 ===")
    pinecone_memory = ExternalMemory(
        embedder_config={
            "provider": "pinecone",
            "config": {
                "api_key": "<YOUR_API_KEY>",
                "environment": "<YOUR_ENV>",
                "index_name": proc_inst_id
            }
        }
    )
    run_flow(f"{proc_inst_id}_pinecone", todos, memory=pinecone_memory)
