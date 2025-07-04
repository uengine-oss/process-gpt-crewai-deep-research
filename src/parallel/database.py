import os
import json
import asyncio
from contextvars import ContextVar
from typing import Optional, List, Dict, Any, Tuple
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# DB 설정 보관용 ContextVar
db_config_var = ContextVar('db_config', default={})

def initialize_db():
    """환경변수 로드 및 DB 설정 초기화"""
    if os.getenv("ENV") != "production":
        load_dotenv()
    db_config = {
        "dbname":   os.getenv("DB_NAME"),
        "user":     os.getenv("DB_USER"),
        "password": os.getenv("DB_PASSWORD"),
        "host":     os.getenv("DB_HOST"),
        "port":     os.getenv("DB_PORT"),
    }
    db_config_var.set(db_config)


def _get_connection():
    """ContextVar에 저장된 설정으로 새 커넥션 생성"""
    cfg = db_config_var.get()
    conn = psycopg2.connect(**cfg)
    conn.autocommit = False
    return conn

async def fetch_pending_task(limit: int = 1) -> Optional[Dict[str, Any]]:
    """
    새로운 작업 조회(SELECT FOR UPDATE SKIP LOCKED)
    가져오는 즉시 draft_status를 'RUNNING'으로 업데이트하여 중복 처리 방지
    """
    def _sync():
        conn = _get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT * FROM todolist
            WHERE status = 'IN_PROGRESS'
              AND (
                (agent_mode = 'DRAFT'
                  AND (draft IS NULL OR draft::text = '' OR draft::text = 'EMPTY')
                  AND draft_status IS NULL
                )
                OR draft_status = 'FB_REQUESTED'
              )
            ORDER BY start_date ASC
            LIMIT %s
            FOR UPDATE SKIP LOCKED
        """, (limit,))
        row = cur.fetchone()
        if not row:
            cur.close()
            conn.close()
            return None
        # 중복 처리 방지: RUNNING 상태로 변경
        cur.execute(
            "UPDATE todolist SET draft_status = 'RUNNING' WHERE id = %s",
            (row['id'],)
        )
        conn.commit()
        return { 'row': row, 'connection': conn, 'cursor': cur }
    return await asyncio.to_thread(_sync)

async def fetch_done_data(proc_inst_id: Optional[str]) -> Tuple[List[Any], List[Any]]:
    """
    완료된 작업들의 output(또는 draft) 및 feedback 리스트 조회
    """
    def _sync():
        outputs: List[Any] = []
        feedbacks: List[Any] = []
        if not proc_inst_id:
            return outputs, feedbacks
        conn = _get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT status, output, feedback, draft FROM todolist
               WHERE proc_inst_id = %s
                 AND (
                   (status = 'DONE' AND output IS NOT NULL)
                   OR (status = 'IN_PROGRESS' AND feedback IS NOT NULL)
                 )
               ORDER BY start_date ASC""",
            (proc_inst_id,)
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()
        for r in rows:
            if isinstance(r, dict):
                status = r.get('status')
                feedback = r.get('feedback')
                # IN_PROGRESS + feedback 있는 경우 draft에서 reports만 추출
                if status == 'IN_PROGRESS' and feedback is not None:
                    draft = r.get('draft')
                    if isinstance(draft, str):
                        try:
                            parsed = json.loads(draft)
                            outputs.append(parsed.get('reports'))
                        except Exception:
                            outputs.append(draft)
                    elif isinstance(draft, dict):
                        outputs.append(draft.get('reports'))
                    else:
                        outputs.append(draft)
                else:
                    outputs.append(r.get('output'))
                feedbacks.append(feedback)
            else:
                # row tuple: (status, output, feedback, draft)
                status = r[0]
                feedback = r[2] if len(r) > 2 else None
                # IN_PROGRESS + feedback 있는 경우 draft에서 reports만 추출
                if status == 'IN_PROGRESS' and feedback is not None:
                    draft = r[3] if len(r) > 3 else None
                    if isinstance(draft, str):
                        try:
                            parsed = json.loads(draft)
                            outputs.append(parsed.get('reports'))
                        except Exception:
                            outputs.append(draft)
                    elif isinstance(draft, dict):
                        outputs.append(draft.get('reports'))
                    else:
                        outputs.append(draft)
                else:
                    outputs.append(r[1] if len(r) > 1 else None)
                feedbacks.append(feedback)
        return outputs, feedbacks
    return await asyncio.to_thread(_sync)

async def save_task_result(todo_id: int, result: Any, final: bool = False) -> None:
    """주어진 todo_id로 행을 조회해 결과 저장. final=True일 때 draft_status를 'COMPLETED'로 업데이트"""
    def _sync():
        conn = _get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        payload = json.dumps(result, ensure_ascii=False)
        
        # 기존 agent_mode 조회
        cur.execute(
            "SELECT agent_mode FROM todolist WHERE id = %s",
            (todo_id,)
        )
        row = cur.fetchone() or {}
        mode = row.get('agent_mode')

        # 중간 저장(final=False): draft만 업데이트
        if not final:
            cur.execute(
                "UPDATE todolist SET draft=%s WHERE id = %s",
                (payload, todo_id)
            )
        else:
            # 최종 저장(final=True): draft_status 및 필요시 output, status 업데이트
            if mode == 'COMPLETE':
                cur.execute(
                    "UPDATE todolist SET output=%s, draft=%s, status='SUBMITTED', draft_status='COMPLETED' WHERE id = %s",
                    (payload, payload, todo_id)
                )
            else:
                cur.execute(
                    "UPDATE todolist SET draft=%s, draft_status='COMPLETED' WHERE id = %s",
                    (payload, todo_id)
                )
        conn.commit()
        cur.close()
        conn.close()
    await asyncio.to_thread(_sync)

async def fetch_task_status(todo_id: int) -> Optional[str]:
    """지정된 todo_id의 draft_status를 조회합니다."""
    def _sync():
        conn = _get_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            "SELECT draft_status FROM todolist WHERE id = %s",
            (todo_id,)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row.get('draft_status') if row else None
    return await asyncio.to_thread(_sync) 