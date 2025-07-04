import json
from typing import Dict, List, Any, Optional
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field
from ..settings.crew_config_manager import CrewConfigManager
from ..crews.report_crew.DynamicReportCrew import DynamicReportCrew
from ..settings.crew_event_logger import CrewAIEventLogger
import traceback
import asyncio
import re
from ..agents_repository import AgentsRepository
from ..database import save_task_result

# Phase 모델: 각 단계의 폼 리스트와 전략을 담는 Pydantic 모델
class Phase(BaseModel):
    forms: List[Dict[str, Any]] = Field(default_factory=list)
    strategy: Optional[str] = None

# ExecutionPlan 모델: report/slide/text 단계별 Phase를 담는 모델
class ExecutionPlan(BaseModel):
    report_phase: Phase = Field(default_factory=Phase)    # 리포트 폼 목록
    slide_phase: Phase = Field(default_factory=Phase)     # 슬라이드 폼 목록
    text_phase: Phase = Field(default_factory=Phase)      # 텍스트 폼 목록

# MultiFormatState 모델: 플로우 상태를 저장하는 Pydantic 모델
class MultiFormatState(BaseModel):
    topic: str = ""
    user_info: List[Dict[str, Any]] = Field(default_factory=list)
    form_types: List[Dict[str, Any]] = Field(default_factory=list)
    execution_plan: Optional[ExecutionPlan] = None
    report_sections: Dict[str, List[Dict[str, Any]]] = Field(default_factory=dict)   # 리포트별 섹션 리스트
    section_contents: Dict[str, Dict[str, str]] = Field(default_factory=dict)        # 섹션별 보고서 내용
    report_contents: Dict[str, str] = Field(default_factory=dict)                    # 통합 리포트 내용
    slide_contents: Dict[str, str] = Field(default_factory=dict)                     # 슬라이드 내용 (중첩 구조 제거)
    text_contents: Dict[str, Any] = Field(default_factory=dict)                      # 텍스트 내용 (중첩 구조 제거)
    todo_id: Optional[int] = None                                                    # ToadList 작업 ID
    proc_inst_id: Optional[str] = None                                               # 프로세스 인스턴스 ID
    form_id: Optional[str] = None                                                    # 폼 ID
    agent_info: Optional[List[Dict[str, Any]]] = Field(default_factory=list)         # 에이전트 정보
    previous_context: str = ""                                                       # 요약된 이전 완료 outputs

# JSON 코드 블록을 처리하고 backtick을 제거하는 헬퍼 함수
def _clean_json_input(raw: Any) -> str:
    text = raw or ""
    if not isinstance(text, str):
        return text
    # ```json
    m = re.search(r"```(?:json)?[\r\n]+(.*?)[\r\n]+```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    # 코드 블록 전체 backtick 제거
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.split("\n")
        return "\n".join(lines[1:-1])
    return text

# MultiFormatFlow: 섹션, 리포트, 슬라이드, 텍스트 생성 플로우 구현
class MultiFormatFlow(Flow[MultiFormatState]):
    def __init__(self):
        # Flow 기본 생성자 호출
        super().__init__()
        self.cm = CrewConfigManager()
        self.event_logger = CrewAIEventLogger()

    @start()
    # create_execution_plan: AI를 이용해 실행 계획 생성 (-> ExecutionPlan 저장)
    async def create_execution_plan(self) -> ExecutionPlan:
        # 1) ExecutionPlanningCrew 생성 및 kickoff
        try:
            crew = self.cm.create_execution_planning_crew()
            out = await crew.kickoff_async(inputs={
                "topic": self.state.topic,
                "form_types": getattr(self.state, 'form_types', []),
                "todo_id": self.state.todo_id,
                "proc_inst_id": self.state.proc_inst_id
            })
            # 2) AI 응답 파싱
            # raw 텍스트에서 JSON 코드 블록 제거 후 파싱
            raw_text = getattr(out, 'raw', out) or ""
            cleaned = _clean_json_input(raw_text)
            # 3) 상태에 실행 계획 저장 (Pydantic Phase alias 사용)
            plan_dict = json.loads(cleaned).get('execution_plan', {})
            self.state.execution_plan = ExecutionPlan.parse_obj(plan_dict)
        except Exception as e:
            print(f"❌ [Error][create_execution_plan] 실행 계획 생성 오류: {e}")
            print(traceback.format_exc())
            raise
        return self.state.execution_plan

    @listen("create_execution_plan")
    # generate_and_merge_report_sections: 섹션별 리포트 생성 및 결과 병합 (-> Dict[보고서][섹션: 내용])
    async def generate_and_merge_report_sections(self) -> Dict[str, Dict[str, str]]:
        # 섹션별 리포트 생성 및 결과 병합
        try:
            for report_form in self.state.execution_plan.report_phase.forms:
                report_key = report_form.get('key')
                # 1) TOC 및 에이전트 매칭
                # 사용 가능한 에이전트 목록 조회
                agents = await AgentsRepository().get_all_agents()
                crew = self.cm.create_agent_matching_crew()
                # 이전 컨텍스트 조회
                prev_context = self.state.previous_context
                out = await crew.kickoff_async(inputs={
                    "topic": self.state.topic,
                    "user_info": self.state.user_info,
                    "agent_info": self.state.agent_info,
                    "previous_context": prev_context,
                    "available_agents": agents,
                    "todo_id": self.state.todo_id,
                    "proc_inst_id": self.state.proc_inst_id
                })
                # 섹션 JSON 파싱 (코드 블록 처리)
                raw_text = getattr(out, 'raw', out) or ""
                cleaned = _clean_json_input(raw_text)
                sections = json.loads(cleaned)
                self.state.report_sections[report_key] = sections
                self.state.section_contents[report_key] = {}
                # 2) 섹션별 리포트 생성 및 중간 저장
                await self._process_report_sections(report_key, sections)
                # 3) 섹션 병합 작업 시작 이벤트 발행
                self.event_logger.emit_event(
                    event_type="task_started",
                    data={
                        "role": "리포트 통합 전문가",
                        "goal": f"리포트의 각 섹션을 하나의 완전한 문서로 병합하여 일관성 있는 최종 리포트를 생성합니다.",
                        "agent_profile": ""
                    },
                    job_id=f"merge-{report_key}",
                    crew_type="report",
                    todo_id=self.state.todo_id,
                    proc_inst_id=self.state.proc_inst_id
                )
                
                # 섹션 결과 병합 (report_sections 순서 유지)
                ordered_titles = [sec_item.get('toc', {}).get('title', 'unknown') for sec_item in self.state.report_sections[report_key]]
                merged_report = "\n\n---\n\n".join([
                    self.state.section_contents[report_key][t]
                    for t in ordered_titles
                    if t in self.state.section_contents[report_key]
                ])
                self.state.report_contents[report_key] = merged_report
                
                # 섹션 병합 작업 완료 이벤트 발행
                self.event_logger.emit_event(
                    event_type="task_completed",
                    data={
                        "final_result": merged_report
                    },
                    job_id=f"merge-{report_key}",
                    crew_type="report",
                    todo_id=self.state.todo_id,
                    proc_inst_id=self.state.proc_inst_id
                )
                
        except Exception as e:
            print(f"❌ [Error][generate_and_merge_report_sections] {e}")
            print(traceback.format_exc())
            raise
        return self.state.section_contents

    # _run_dynamic_report: 단일 섹션 리포트 생성 및 에러 전파
    async def _run_dynamic_report(self, sec: Dict[str, Any]) -> str:
        # 단일 섹션 리포트 생성 및 에러 전파
        title = sec.get('toc', {}).get('title', 'unknown')
        try:
            # DynamicReportCrew 생성 및 kickoff
            prev_context = self.state.previous_context
            crew = DynamicReportCrew(sec, self.state.topic, prev_context)
            out = await crew.create_crew().kickoff_async(inputs={
                "todo_id": self.state.todo_id,
                "proc_inst_id": self.state.proc_inst_id
            })
            return getattr(out, 'raw', out)
        except Exception as e:
            print(f"❌ [Error][_run_dynamic_report] section={title}, error={e}")
            print(traceback.format_exc())
            raise

    async def _process_report_sections(self, report_key: str, sections: List[Dict[str, Any]]) -> None:
        # 비동기로 각 섹션을 생성하고, 완료 시마다 중간 결과를 병합 후 DB 저장
        # 1) Task 생성 및 섹션 매핑
        tasks_list = [asyncio.create_task(self._run_dynamic_report(sec)) for sec in sections]
        sec_map = {task: sec for task, sec in zip(tasks_list, sections)}
        pending = set(tasks_list)
        # 2) 완료된 순서대로 처리
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                sec = sec_map[task]
                title = sec.get('toc', {}).get('title', 'unknown')
                try:
                    result = task.result()
                    self.state.section_contents[report_key][title] = result
                except Exception as e:
                    print(f"❌ [Error][_process_report_sections] report={report_key}, section={title} error={e}")
                    self.state.section_contents[report_key][title] = f"Error: {e}"
                # 3) 순서 보장하여 현재까지 처리된 섹션 병합
                ordered_titles = [s.get('toc', {}).get('title', 'unknown') for s in sections]
                merged = "\n\n---\n\n".join(
                    [self.state.section_contents[report_key][t] for t in ordered_titles if t in self.state.section_contents[report_key]]
                )
                self.state.report_contents[report_key] = merged
                # 4) DB에 중간 결과 저장
                if self.state.todo_id is not None:
                    await save_task_result(self.state.todo_id, self.state.report_contents)
                else:
                    print("⚠️ todo_id가 None입니다. 중간 결과 저장을 건너뜁니다.")
        # helper 종료

    @listen("generate_and_merge_report_sections")
    # generate_slides_from_reports: report_contents 기반 슬라이드 생성 (-> Dict[슬라이드키: 내용])
    async def generate_slides_from_reports(self) -> Dict[str, str]:
        # 슬라이드 생성 및 결과 저장 - 중첩 구조 제거
        try:
            for report_key, merged_content in self.state.report_contents.items():
                # prepare slide tasks
                for slide_form in self.state.execution_plan.slide_phase.forms:
                    if report_key in slide_form.get('dependencies', []):
                        slide_key = slide_form['key']
                        crew = self.cm.create_slide_crew()
                        result = await crew.kickoff_async(inputs={
                            'report_content': merged_content,
                            'user_info': self.state.user_info,
                            'todo_id': self.state.todo_id,
                            'proc_inst_id': self.state.proc_inst_id
                        })
                        # 직접 slide_contents에 저장 (중첩 구조 제거)
                        self.state.slide_contents[slide_key] = getattr(result, 'raw', result)
        except Exception as e:
            print(f"❌ [Error][generate_slides_from_reports] {e}")
            print(traceback.format_exc())
            raise
        return self.state.slide_contents

    @listen("generate_slides_from_reports")
    # generate_texts_from_reports: report_contents 기반 텍스트 생성 (-> Dict[텍스트키: 내용])
    async def generate_texts_from_reports(self) -> Dict[str, Any]:
        # 텍스트 생성 및 결과 저장 - 보고서 내용을 기반으로 생성
        try:
            for report_key, merged_content in self.state.report_contents.items():
                # prepare text tasks
                for text_form in self.state.execution_plan.text_phase.forms:
                    if report_key in text_form.get('dependencies', []):
                        form_key = text_form['key']
                        crew = self.cm.create_form_crew()
                        result = await crew.kickoff_async(inputs={
                            'report_content': merged_content,  # 보고서 내용 사용 (슬라이드 내용 아님)
                            'topic': self.state.topic,
                            'user_info': self.state.user_info,
                            'todo_id': self.state.todo_id,
                            'proc_inst_id': self.state.proc_inst_id
                        })
                        # 직접 text_contents에 저장 (중첩 구조 제거)
                        self.state.text_contents[form_key] = getattr(result, 'raw', result)
        except Exception as e:
            print(f"❌ [Error][generate_texts_from_reports] {e}")
            print(traceback.format_exc())
            raise
        return self.state.text_contents

    @listen("generate_texts_from_reports")
    # compile_and_output_results: 최종 통계 출력 및 결과 저장 및 DB 저장
    async def compile_and_output_results(self) -> None:
        # 최종 통계 출력 및 결과 반환
        try:
            print("\n" + "="*60)
            print("🎉 MULTI-FORMAT GENERATION COMPLETED!")
            print("="*60)
            # 요약 통계 - 중첩 구조 제거에 따른 계산 방법 변경
            report_count = len(self.state.report_contents)
            slide_count = len(self.state.slide_contents)  # 직접 길이 계산
            form_count = len(self.state.text_contents)    # 직접 길이 계산
            print(f"📊 처리 결과: 리포트 {report_count}개, 슬라이드 {slide_count}개, 폼 {form_count}개")

            # DB에 결과 저장
            result = {
                'reports': self.state.report_contents,
                'slides': self.state.slide_contents,
                'texts': self.state.text_contents
            }
            if self.state.todo_id is not None:
                # 최종 호출: final flag 전달하여 draft_status 완료로 변경
                await save_task_result(self.state.todo_id, result, final=True)
            else:
                print("⚠️ todo_id가 None입니다. DB 저장을 건너뜁니다.")

            self.event_logger.emit_event(
                    event_type="crew_completed",
                    data={},
                    job_id=f"CREW_FINISHED",
                    crew_type="crew",
                    todo_id=self.state.todo_id,
                    proc_inst_id=self.state.proc_inst_id
            )
                
        except Exception as e:
            print(f"❌ [Error][compile_and_output_results] error={e}")
            print(traceback.format_exc())
            raise