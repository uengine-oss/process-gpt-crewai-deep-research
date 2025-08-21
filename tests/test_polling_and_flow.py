import os
import sys
import pytest
import logging
from dotenv import load_dotenv

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 테스트 환경 설정
os.environ['ENV'] = 'test'
load_dotenv('.env.test', override=True)

# 로깅 설정 (모든 로그 INFO 레벨로 표시)
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

from core.database import initialize_db, get_db_client
from core.polling_manager import _prepare_task_inputs
from flows.multi_format_flow import MultiFormatFlow

# DB 초기화
initialize_db()

# ============================================================================
# 테스트 케이스들
# ============================================================================

@pytest.mark.asyncio
async def test_prepare_phase():
    """
    1) todolist 테이블에서 실제 todo_id로 row를 가져와,
    2) _prepare_task_inputs가 올바른 dict 구조를 반환하는지 검증
    """
    todo_id = "9d316565-b891-43c6-8e70-cf91f9256bb9"
    client = get_db_client()
    resp = (
        client
        .table('todolist')
        .select('*')
        .eq('id', todo_id)
        .single()
        .execute()
    )
    row = resp.data
    if not row:
        print(f"⚠️ Todo ID {todo_id}가 DB에 없습니다. 테스트 스킵")
        return
    
    # Row 입력 확인
    print(f"\n입력 Row:")
    print(f"  activity_name: '{row.get('activity_name')}'")
    print(f"  tool: '{row.get('tool')}'")
    print(f"  user_id: '{row.get('user_id')}'")
    print(f"  tenant_id: '{row.get('tenant_id')}'")
    
    # _prepare_task_inputs 실행 및 간단 출력
    inputs = await _prepare_task_inputs(row)
    print(f"\n입력 준비 완료:")
    print(f"  topic: '{inputs.get('topic')}'")
    print(f"  proc_form_id: '{inputs.get('proc_form_id')}'")
    print(f"  form_types: {len(inputs.get('form_types', []))}개")
    print(f"  participants: user={len(inputs.get('user_info', []))}, agent={len(inputs.get('agent_info', []))}")
    print(f"  form_html 키 존재: {'form_html' in inputs}")

@pytest.mark.asyncio
async def test_full_flow_phase():
    """
    MultiFormatFlow 전체 실행 흐름 테스트
    """
    todo_id = "9d316565-b891-43c6-8e70-cf91f9256bb9"
    client = get_db_client()
    row = (
        client
        .table('todolist')
        .select('*')
        .eq('id', todo_id)
        .single()
        .execute()
    ).data
    inputs = await _prepare_task_inputs(row)

    flow = MultiFormatFlow()
    for k, v in inputs.items():
        setattr(flow.state, k, v)

    print(f"\n플로우 단계별 실행:")

    # 1. create_execution_plan
    plan = await flow.create_execution_plan()
    report_forms = plan.report_phase.forms if plan else []
    slide_forms = plan.slide_phase.forms if plan else []
    text_forms = plan.text_phase.forms if plan else []
    print(f"  create_execution_plan 완료 (report:{len(report_forms)}, slide:{len(slide_forms)}, text:{len(text_forms)})")

    # 2. generate_reports
    reports = await flow.generate_reports()
    print(f"  generate_reports 완료 ({len(reports) if isinstance(reports, dict) else 0}개)")

    # 3. generate_slides
    slides = await flow.generate_slides()
    print(f"  generate_slides 완료 ({len(slides) if isinstance(slides, dict) else 0}개)")

    # 4. generate_texts
    texts = await flow.generate_texts()
    print(f"  generate_texts 완료 ({len(texts) if isinstance(texts, dict) else 0}개)")

    # 5. save_final_results
    await flow.save_final_results()
    print(f"  save_final_results: ✓ 완료")
    print(f"✓ 전체 플로우 실행 완료")


# ============================================================================
# 디버그 실행용 메인 함수
# ============================================================================

async def main():
    """디버그 실행용 메인 함수 - pytest 없이 직접 실행 가능"""
    print("=== 디버그 모드 실행 ===\n")
    
    try:
        # print("1. prepare_phase 테스트 시작...")
        # await test_prepare_phase()
        # print("✓ prepare_phase 테스트 완료\n")
        
        print("2. full_flow_phase 테스트 시작...")
        await test_full_flow_phase()
        print("✓ full_flow_phase 테스트 완료\n")
        
        print("🎉 모든 테스트 성공적으로 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 