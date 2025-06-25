"""
CrewAI Event Logger - Task/Agent 이벤트 전용 (Supabase 스키마 호환)
"""

import os
import uuid
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Set, Any as TypeAny
import logging
import re

from dotenv import load_dotenv
from ..context_manager import crew_type_var, todo_id_var, proc_id_var

# .env 파일 로드
load_dotenv()


# 로깅 설정
logger = logging.getLogger(__name__)

# Supabase client availability
from supabase import create_client, Client
SUPABASE_AVAILABLE = True

class CrewAIEventLogger:
    """
    CrewAI 이벤트 로깅 시스템 - Task/Agent 전용, Supabase 스키마 호환
    
    특징:
    - Task와 Agent 이벤트만 기록 (Crew 이벤트 완전 제외)
    - Supabase 스키마 완벽 호환 (id, job_id, type, data, timestamp)
    - 중복 이벤트 자동 제거
    - 단일 로그 파일 생성
    """
    
    # === Initialization ===
    def __init__(self):
        """이벤트 로거 초기화 (Supabase 로깅만 활성화)"""
        # Supabase 클라이언트 초기화
        self.supabase_client = self._init_supabase()
        self.log_file = None
        logger.info(f"🎯 CrewAI Event Logger 초기화 - Supabase 로깅만 활성화")
        print(f"   - Supabase: {'✅' if self.supabase_client else '❌'}")

    def _init_supabase(self) -> Optional[Client]:
        """Supabase 클라이언트 초기화"""
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY") 
        
        if not url or not key:
            logger.warning("⚠️ Supabase 자격증명 누락 - 로깅 비활성화")
            return None
        
        try:
            client = create_client(url, key)
            logger.info("✅ Supabase 백엔드 연결됨")
            return client
        except Exception as e:
            logger.error(f"❌ Supabase 연결 실패: {e}")
            return None

    # === Job ID Generation ===
    def _generate_job_id(self, event_obj: TypeAny, source: TypeAny) -> str:
        # 항상 task.id 사용
        if hasattr(event_obj, 'task') and hasattr(event_obj.task, 'id'):
            return str(event_obj.task.id)
        if source and hasattr(source, 'task') and hasattr(source.task, 'id'):
            return str(source.task.id)
        return 'unknown'


    # === Event Data Extraction ===
    def _extract_event_data(self, event_obj: TypeAny, source: Optional[TypeAny] = None) -> Dict[str, Any]:
        event_type = event_obj.type
        try:
            if event_type == "task_started":
                role = getattr(event_obj.task.agent, 'role', 'Unknown')
                goal = getattr(event_obj.task.agent, 'goal', 'Unknown')
                profile = getattr(event_obj.task.agent, 'profile', None)
                return {"role": role, "goal": goal, "agent_profile": profile}
            elif event_type == "task_completed":
                final_result = getattr(event_obj, 'output', 'Completed')
                return {"final_result": str(final_result)}
            elif event_type.startswith('tool_'):
                tool_name = getattr(event_obj, 'tool_name', None)
                tool_args = getattr(event_obj, 'tool_args', None)
                query = None
                if tool_args:
                    try:
                        args_dict = json.loads(tool_args)
                        query = args_dict.get('query')
                    except Exception:
                        query = None
                return {"tool_name": tool_name, "query": query}
            else:
                return {"info": f"Event type: {event_type}"}
        except Exception as e:
            logger.error(f"Error extracting event data: {e}")
            return {"error": f"Failed to extract data: {str(e)}"}

    # === Backend Writing ===
    def _write_to_backends(self, event_record: Dict[str, Any]) -> None:
        """Supabase와 파일에 기록 (동기화 처리로 누락 방지)"""
        # Supabase 기록
        if self.supabase_client:
            try:
                # 🔧 안전한 JSON 직렬화: 모든 객체를 문자열로 변환
                def safe_serialize(obj):
                    """모든 객체를 JSON 직렬화 가능한 형태로 변환"""
                    if hasattr(obj, 'raw'):  # TaskOutput 객체
                        return str(obj.raw)
                    elif hasattr(obj, '__dict__'):  # 일반 객체
                        return str(obj)
                    else:
                        return str(obj)
                
                serializable_record = json.loads(json.dumps(event_record, default=safe_serialize))
                print("serializable_record: ", json.dumps(serializable_record, ensure_ascii=False, indent=2))
                self.supabase_client.table("events").insert(serializable_record).execute()
            except Exception as e:
                logger.error(f"❌ Supabase 저장 실패: {e}")
                print(f"❌ Supabase 저장 실패: {e}")
                # 디버깅용: 문제가 되는 데이터 구조 출력
                print(f"🔍 문제 데이터: {type(event_record.get('data', {}))}")
                for key, value in event_record.get('data', {}).items():
                    print(f"🔍 data.{key}: {type(value)} = {str(value)[:100]}...")


    # === Event Processing Entry Point ===
    def on_event(self, event_obj: TypeAny, source: Optional[TypeAny] = None) -> None:
        """Task와 Tool 이벤트 처리 (Agent/Crew 이벤트는 완전히 제외)"""
        try:
            # task, tool 이벤트만 처리
            et = event_obj.type
            if et not in ["task_started", "task_completed", "tool_usage_started", "tool_usage_finished"]:
                return
            
            # job_id 생성 및 데이터 추출
            job_id = self._generate_job_id(event_obj, source)
            event_data = self._extract_event_data(event_obj, source)
            
            # ContextVar에서 현재 크루 컨텍스트 가져오기
            crew_type = crew_type_var.get()
            todo_id = todo_id_var.get()
            proc_inst_id = proc_id_var.get()
            
            # 🔧 data 필드를 안전하게 직렬화 가능한 형태로 변환
            safe_data = {}
            for key, value in event_data.items():
                try:
                    # TaskOutput 객체 처리
                    if hasattr(value, 'raw'):
                        safe_data[key] = str(value.raw)
                    # 기타 복잡한 객체 처리
                    elif hasattr(value, '__dict__') and not isinstance(value, (str, int, float, bool, type(None))):
                        safe_data[key] = str(value)
                    else:
                        safe_data[key] = value

                    print("safe_data: ", json.dumps(safe_data, ensure_ascii=False, indent=2))

                except Exception as e:
                    logger.warning(f"Data 직렬화 실패 ({key}): {e}")
                    safe_data[key] = f"[직렬화 실패: {type(value).__name__}]"
            
            # 🆕 단순화된 스키마로 레코드 생성
            event_record = {
                "id": str(uuid.uuid4()),
                "job_id": job_id,
                "todo_id": todo_id,              # todolist 항목 ID
                "proc_inst_id": proc_inst_id,    # 프로세스 인스턴스 ID
                "event_type": event_obj.type,
                "crew_type": crew_type,
                "data": safe_data,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            # 백엔드에 기록
            self._write_to_backends(event_record)
            
            # 출신 정보 포함한 상세한 콘솔 출력
            tool_info = f" ({safe_data.get('tool_name', '')})" if event_obj.type.startswith('tool_') else ""
            print(f"📝 [{event_obj.type}]{tool_info} [{crew_type}] {job_id[:8]} → Supabase: {'✅' if self.supabase_client else '❌'}")
            
        except Exception as e:
            logger.error(f"❌ 이벤트 처리 실패 ({getattr(event_obj, 'type', 'unknown')}): {e}")

    # === Custom Event Emission ===
    def emit_event(self, event_type: str, data: Dict[str, Any], job_id: str = None, crew_type: str = None, todo_id: str = None, proc_inst_id: str = None) -> None:
        """
        커스텀 이벤트 발행 (모든 event_type에 재사용)
        Args:
            event_type: 이벤트 타입 (e.g. 'task_started')
            data: 이벤트 데이터 사전
            job_id: 작업 식별자, 기본값은 event_type
            crew_type: 크루 타입, 기본값은 ContextVar에서 가져옴
            todo_id: 투두 ID, 기본값은 ContextVar에서 가져옴
            proc_inst_id: 프로세스 인스턴스 ID, 기본값은 ContextVar에서 가져옴
        """
        # ContextVar에서 크루 컨텍스트 가져오기 (인자가 없을 경우)
        crew_type = crew_type or crew_type_var.get()
        todo_id = todo_id or todo_id_var.get()
        proc_inst_id = proc_inst_id or proc_id_var.get()
        record = {
            'id': str(uuid.uuid4()),
            'job_id': job_id,
            'todo_id': todo_id,
            'proc_inst_id': proc_inst_id,
            'event_type': event_type,
            'crew_type': crew_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
        }
        self._write_to_backends(record)
        print(f"📝 [{event_type}] [{crew_type}] {record['job_id'][:8]} → Supabase: {'✅' if self.supabase_client else '❌'}") 