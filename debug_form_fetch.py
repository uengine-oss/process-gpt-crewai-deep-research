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

from src.parallel.database import initialize_db, get_db_client, fetch_form_types

async def debug_form_fetch(todo_id: str):
    """실제 todo ID로 폼 조회 프로세스 디버깅"""
    
    print(f"🔍 Todo ID로 폼 조회 테스트 시작: {todo_id}")
    print("=" * 60)
    
    try:
        # 1. DB 초기화
        print("1️⃣ DB 초기화 중...")
        initialize_db()
        supabase = get_db_client()
        print("✅ DB 초기화 완료")
        
        # 2. todolist 테이블에서 해당 todo 조회
        print(f"\n2️⃣ todolist 테이블에서 todo 조회: {todo_id}")
        todo_resp = (
            supabase
            .table('todolist')
            .select('id, tool, tenant_id, user_id, activity_name, proc_inst_id, status, draft_status')
            .eq('id', todo_id)
            .single()
            .execute()
        )
        
        if not todo_resp.data:
            print(f"❌ Todo ID {todo_id}를 찾을 수 없습니다.")
            return
            
        todo_data = todo_resp.data
        print(f"✅ Todo 데이터 조회 완료:")
        for key, value in todo_data.items():
            print(f"   {key}: {value}")
        
        # 3. tool과 tenant_id 추출
        print(f"\n3️⃣ 필드 추출")
        tool_val = todo_data.get('tool', '')
        tenant_id = str(todo_data.get('tenant_id', ''))
        
        print(f"   tool: '{tool_val}'")
        print(f"   tenant_id: '{tenant_id}'")
        
        # 4. form_id 추출 로직 테스트
        print(f"\n4️⃣ form_id 추출")
        form_id = tool_val[12:] if tool_val.startswith('formHandler:') else tool_val
        print(f"   원본 tool: '{tool_val}'")
        print(f"   추출된 form_id: '{form_id}'")
        
        # 5. form_def 테이블에서 폼 정보 조회 (단계별)
        print(f"\n5️⃣ form_def 테이블 조회 테스트")
        
        # 5-1. form_id만으로 검색
        print(f"   5-1. form_id만으로 검색: '{form_id}'")
        form_only_resp = (
            supabase
            .table('form_def')
            .select('id, tenant_id, fields_json')
            .eq('id', form_id)
            .execute()
        )
        print(f"      결과 개수: {len(form_only_resp.data) if form_only_resp.data else 0}")
        if form_only_resp.data:
            for i, record in enumerate(form_only_resp.data):
                print(f"      [{i}] id: {record.get('id')}, tenant_id: {record.get('tenant_id')}")
        
        # 5-2. tenant_id만으로 검색 (최대 3개)
        print(f"   5-2. tenant_id만으로 검색: '{tenant_id}'")
        tenant_only_resp = (
            supabase
            .table('form_def')
            .select('id, tenant_id, fields_json')
            .eq('tenant_id', tenant_id)
            .limit(3)
            .execute()
        )
        print(f"      결과 개수: {len(tenant_only_resp.data) if tenant_only_resp.data else 0}")
        if tenant_only_resp.data:
            for i, record in enumerate(tenant_only_resp.data):
                print(f"      [{i}] id: {record.get('id')}, tenant_id: {record.get('tenant_id')}")
        
        # 5-3. 두 조건 모두로 검색 (배열 방식)
        print(f"   5-3. 두 조건 모두로 검색 (배열 방식)")
        both_resp = (
            supabase
            .table('form_def')
            .select('id, tenant_id, fields_json')
            .eq('id', form_id)
            .eq('tenant_id', tenant_id)
            .execute()
        )
        print(f"      결과 개수: {len(both_resp.data) if both_resp.data else 0}")
        if both_resp.data:
            for i, record in enumerate(both_resp.data):
                print(f"      [{i}] id: {record.get('id')}, tenant_id: {record.get('tenant_id')}")
                fields_json = record.get('fields_json')
                if fields_json:
                    print(f"          fields_json: {fields_json}")
                else:
                    print(f"          fields_json: None 또는 빈 값")
        
        # 5-4. 두 조건 모두로 검색 (single 방식)
        print(f"   5-4. 두 조건 모두로 검색 (single 방식)")
        try:
            single_resp = (
                supabase
                .table('form_def')
                .select('id, tenant_id, fields_json')
                .eq('id', form_id)
                .eq('tenant_id', tenant_id)
                .single()
                .execute()
            )
            print(f"      single 결과: {single_resp.data}")
            if single_resp.data:
                fields_json = single_resp.data.get('fields_json')
                print(f"      fields_json: {fields_json}")
        except Exception as e:
            print(f"      single 방식 오류: {e}")
        
        # 6. 실제 fetch_form_types 함수 호출
        print(f"\n6️⃣ 실제 fetch_form_types 함수 호출")
        try:
            proc_form_id, form_types = await fetch_form_types(tool_val, tenant_id)
            print(f"✅ fetch_form_types 결과:")
            print(f"   proc_form_id: {proc_form_id}")
            print(f"   form_types: {form_types}")
        except Exception as e:
            print(f"❌ fetch_form_types 오류: {e}")
            import traceback
            print(f"   상세 정보: {traceback.format_exc()}")
        
    except Exception as e:
        print(f"❌ 전체 프로세스 오류: {e}")
        import traceback
        print(f"상세 정보: {traceback.format_exc()}")

async def main():
    todo_id = "d2fe5208-019a-4d2d-9803-d1d09ac551d2"
    await debug_form_fetch(todo_id)

if __name__ == "__main__":
    asyncio.run(main()) 