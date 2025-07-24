#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import sys
import os

# 프로젝트 루트를 import 경로에 추가
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "src")
    )
)

from src.parallel.database import initialize_db, get_db_client
from src.parallel.polling_manager import _prepare_task_inputs

async def test_real_logic(todo_id: str):
    """실제 로직을 사용해서 특정 todo ID로 테스트"""
    
    print(f"🔍 실제 로직 테스트 시작: {todo_id}")
    print("=" * 60)
    
    try:
        # 1. DB 초기화
        print("1️⃣ DB 초기화 중...")
        initialize_db()
        supabase = get_db_client()
        print("✅ DB 초기화 완료")
        
        # 2. todolist 테이블에서 해당 todo 조회 (실제 데이터 구조 확인)
        print(f"\n2️⃣ todolist 테이블에서 todo 조회: {todo_id}")
        todo_resp = (
            supabase
            .table('todolist')
            .select('*')  # 모든 필드 조회
            .eq('id', todo_id)
            .single()
            .execute()
        )
        
        if not todo_resp.data:
            print(f"❌ Todo ID {todo_id}를 찾을 수 없습니다.")
            return
            
        row = todo_resp.data
        print(f"✅ Todo 데이터 조회 완료:")
        for key, value in row.items():
            print(f"   {key}: {value}")
        
        # 3. 실제 _prepare_task_inputs 함수 호출
        print(f"\n3️⃣ 실제 _prepare_task_inputs 함수 호출")
        print("=" * 40)
        
        inputs = await _prepare_task_inputs(row)
        
        print(f"\n✅ _prepare_task_inputs 결과:")
        print("=" * 40)
        for key, value in inputs.items():
            if key == 'form_types':
                print(f"   {key}: {value}")
                print(f"      폼 개수: {len(value) if value else 0}")
                if value:
                    for i, form in enumerate(value):
                        print(f"      [{i}] {form}")
            elif key == 'previous_context':
                context_preview = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                print(f"   {key}: {context_preview}")
            else:
                print(f"   {key}: {value}")
        
        # 4. 특히 form_types 상세 분석
        form_types = inputs.get('form_types', [])
        print(f"\n4️⃣ form_types 상세 분석")
        print("=" * 40)
        
        if not form_types:
            print("❌ form_types가 비어있습니다!")
        else:
            print(f"✅ form_types 개수: {len(form_types)}")
            for i, form_type in enumerate(form_types):
                print(f"   [{i}] {form_type}")
                
                # 타입별 분류
                form_type_value = form_type.get('type', '')
                if form_type_value == 'text':
                    print(f"       → text_phase에 포함될 예정")
                elif form_type_value == 'report':
                    print(f"       → report_phase에 포함될 예정") 
                elif form_type_value == 'slide':
                    print(f"       → slide_phase에 포함될 예정")
                elif form_type_value == 'default':
                    print(f"       → ⚠️ default 타입 - 어떤 phase에도 포함되지 않음!")
                else:
                    print(f"       → ⚠️ 알 수 없는 타입: {form_type_value}")
        
    except Exception as e:
        print(f"❌ 테스트 오류: {e}")
        import traceback
        print(f"상세 정보: {traceback.format_exc()}")

async def main():
    todo_id = "d2fe5208-019a-4d2d-9803-d1d09ac551d2"
    await test_real_logic(todo_id)

if __name__ == "__main__":
    asyncio.run(main()) 