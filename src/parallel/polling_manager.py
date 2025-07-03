import asyncio
import logging
import json
import os
import sys
from typing import Optional, List, Dict, Any, Tuple
from contextvars import ContextVar
from dotenv import load_dotenv

from supabase import create_client, Client

# 필요한 모듈 임포트
from .context_manager import summarize
from .database import initialize_db, fetch_pending_task, fetch_done_data

# ============================================================================
# 공통 설정 및 초기화
# ============================================================================

logger = logging.getLogger("polling_manager")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# 공통 컨텍스트 변수
supabase_client_var = ContextVar('supabase', default=None)

# 글로벌 플래그 & 프로세스 핸들러
current_todo_id: Optional[int] = None
current_process: Optional[asyncio.subprocess.Process] = None
worker_terminated_by_us: bool = False

def initialize_connections():
    """데이터베이스 및 Supabase 연결 초기화"""
    try:
        # DB 설정 초기화 (database.py 관리)
        initialize_db()
        # Supabase 초기화
        if os.getenv("ENV") != "production":
            load_dotenv()
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        supabase: Client = create_client(supabase_url, supabase_key)
        supabase_client_var.set(supabase)
        logger.info("✅ 연결 초기화 완료")
    except Exception as e:
        logger.error(f"❌ 초기화 오류: {e}")


# ============================================================================
# 새 작업 처리 (TodoList Polling)
# ============================================================================

async def process_new_task(bundle: Dict):
    global current_process, worker_terminated_by_us, current_todo_id

    row, conn, cur = bundle['row'], bundle['connection'], bundle['cursor']
    current_todo_id = row['id']
    todo_id = row['id']
    proc_inst_id = row.get('proc_inst_id')

    try:
        logger.info(f"🆕 새 작업 처리 시작: id={todo_id}, proc_inst_id={proc_inst_id}")

        # 1) 이전 컨텍스트 요약
        done_outputs, done_feedbacks = await fetch_done_data(proc_inst_id)
        context_summary = summarize(done_outputs, done_feedbacks)

        # 2) 사용자 & 폼 조회
        participants = await _get_user_or_agent_info(row.get('user_id', ''))
        form_types = await _get_form_types(row.get('tool', ''))

        # 3) 워커에 넘길 inputs 준비
        inputs = {
            "todo_id": todo_id,
            "proc_inst_id": proc_inst_id,
            "topic": row.get('activity_name', ''),
            "previous_context": context_summary,
            "user_info": participants.get('user_info', []),
            "agent_info": participants.get('agent_info', []),
            "form_types": form_types,
        }

        # 4) 워커 프로세스 실행
        worker_terminated_by_us = False
        current_process = await asyncio.create_subprocess_exec(
            sys.executable,
            os.path.join(os.path.dirname(__file__), "worker.py"),
            "--inputs", json.dumps(inputs, ensure_ascii=False),
            # stdout/stderr를 지정하지 않으면 부모(메인) 콘솔에 모두 출력됩니다.
        )
        watch_task = asyncio.create_task(_watch_cancel_status())
        logger.info(f"✅ 워커 시작 (PID={current_process.pid})")

        # 5) 워커 종료 대기
        await current_process.wait()
        if not watch_task.done():
            watch_task.cancel()

        # 6) 종료 결과 로그
        if worker_terminated_by_us:
            logger.info(f"🛑 워커 사용자 중단됨 (PID={current_process.pid})")
        elif current_process.returncode != 0:
            logger.error(f"❌ 워커 비정상 종료 (code={current_process.returncode})")
        else:
            logger.info(f"✅ 워커 정상 종료 (PID={current_process.pid})")

        # 7) 락 해제용 커밋
        conn.commit()

    except Exception as e:
        logger.error(f"❌ process_new_task 오류 (id={todo_id}): {e}")
        conn.rollback()

    finally:
        # 8) 자원 정리
        cur.close()
        conn.close()
        current_process = None
        worker_terminated_by_us = False
        current_todo_id = None


async def _get_user_or_agent_info(user_ids: str) -> Dict:
    """사용자 또는 에이전트 정보 조회 (쉼표로 구분된 여러 ID 지원)"""
    supabase = supabase_client_var.get()
    
    # 쉼표로 구분된 ID들을 분리
    id_list = [id.strip() for id in user_ids.split(',') if id.strip()]
    
    user_info_list = []
    agent_info_list = []
    
    for user_id in id_list:
        # 먼저 users 테이블에서 조회
        user_resp = supabase.table('users').select('username').eq('email', user_id).execute()
        if user_resp.data:
            username = user_resp.data[0]['username']
            user_info_list.append({
                'email': user_id,
                'name': username
            })
            continue
        
        # users 테이블에 없으면 agents 테이블에서 id로 조회
        agent_resp = supabase.table('agents').select('id, name, role, goal, persona, tools, profile').eq('id', user_id).execute()
        if agent_resp.data:
            agent_data = agent_resp.data[0]
            agent_info_list.append({
                'id': agent_data['id'],
                'name': agent_data['name'],
                'role': agent_data['role'],
                'goal': agent_data['goal'],
                'persona': agent_data['persona'],
                'tools': agent_data['tools'],
                'profile': agent_data['profile']
            })
    
    result = {}
    if user_info_list:
        result['user_info'] = user_info_list
    if agent_info_list:
        result['agent_info'] = agent_info_list
    
    print(result)
    return result


async def _get_form_types(tool_val: str) -> List[Dict]:
    """폼 타입 정보 조회 및 정규화"""
    form_id = tool_val[12:] if tool_val.startswith('formHandler:') else tool_val
    
    supabase = supabase_client_var.get()
    resp = supabase.table('form_def').select('fields_json').eq('id', form_id).execute()
    fields_json = resp.data[0].get('fields_json') if resp.data else None
    
    if not fields_json:
        return [{'id': form_id, 'type': 'default'}]
    
    form_types = []
    for field in fields_json:
        field_type = field.get('type', '').lower()
        normalized_type = field_type if field_type in ['report', 'slide'] else 'text'
        form_types.append({
            'id': field.get('key'),
            'type': normalized_type,
            'key': field.get('key'),
            'text': field.get('text', '')
        })
    
    return form_types


# 워커 취소 상태 감시 함수 추가
async def _watch_cancel_status():
    global current_todo_id, current_process, worker_terminated_by_us
    # supabase client를 사용하여 draft_status 조회
    todo_id = current_todo_id
    if todo_id is None:
        return
    supabase: Client = supabase_client_var.get()
    # 주기적으로 draft_status가 CANCELLED인지 확인
    while current_process and current_process.returncode is None and not worker_terminated_by_us:
        await asyncio.sleep(5)
        try:
            resp = supabase.table('todolist').select('draft_status').eq('id', todo_id).single().execute()
            data = resp.data
            draft_status = data.get('draft_status') if isinstance(data, dict) else None
            if draft_status in ('CANCELLED', 'FB_REQUESTED'):
                logger.info(f"🛑 draft_status={draft_status} 감지 (id={todo_id}) → 워커 종료")
                terminate_current_worker()
                break
        except Exception as e:
            logger.error(f"❌ cancel 상태 조회 오류 (id={todo_id}): {e}")


def terminate_current_worker():
    """현재 실행 중인 워커 프로세스에 SIGTERM 전송"""
    global current_process, worker_terminated_by_us
    if current_process and current_process.returncode is None:
        worker_terminated_by_us = True
        current_process.terminate()
        logger.info(f"✅ 워커 프로세스 종료 시그널 전송 (PID={current_process.pid})")
    else:
        logger.warning("⚠️ 종료할 워커 프로세스가 없습니다.")


# ============================================================================
# 통합 Polling 실행부
# ============================================================================

async def start_todolist_polling(interval: int = 7):
    """새 작업 처리 폴링 시작"""
    logger.info("🚀 TodoList 폴링 시작")
    while True:
        try:
            # database.fetch_pending_task 직접 호출
            bundle = await fetch_pending_task()
            if bundle:
                await process_new_task(bundle)
        except Exception as e:
            logger.error(f"❌ TodoList 폴링 오류: {e}")
        await asyncio.sleep(interval)