#!/usr/bin/env python3
"""
줄 단위 diff 분석 - 더 직관적인 변경사항 확인
"""

import difflib
from diff_match_patch import diff_match_patch

def test_line_diff():
    """줄 단위로 직관적인 diff 분석"""
    
    # 복잡한 리포트 예시
    original = """# 2024년 3분기 프로젝트 현황 보고서

## 1. 전체 개요
이번 분기 주요 성과는 다음과 같습니다:
- 백엔드 API 개발 완료 (80%)
- 프론트엔드 UI 구현 (60%)
- 데이터베이스 설계 완료 (90%)

## 2. 주요 이슈
현재 발생한 문제점들:
1. 서버 응답속도 지연 (평균 2.5초)
2. 메모리 사용량 증가
3. 보안 검토 필요

## 3. 향후 일정
- 10월: 성능 최적화
- 11월: 베타 테스트
- 12월: 정식 런칭"""

    modified = """# 2024년 3분기 프로젝트 현황 보고서

## 1. 전체 개요
이번 분기 주요 성과는 다음과 같습니다:
- 백엔드 API 개발 완료 (95%) ← 목표 초과달성
- 프론트엔드 UI 구현 (85%) ← 빠른 진행
- 데이터베이스 설계 완료 (90%)
- 모바일 앱 개발 시작 (20%) ← 신규 추가

## 2. 주요 이슈
현재 발생한 문제점들:
1. 서버 응답속도 개선됨 (평균 1.2초) ← 성능 향상
2. 메모리 사용량 최적화 완료
3. 보안 검토 완료 ← 해결
4. SSL 인증서 갱신 필요 ← 새로운 이슈

## 2.5 신규 개발 항목
- React Native 기반 모바일 앱
- 실시간 알림 시스템
- 사용자 권한 관리 시스템

## 3. 향후 일정
- 10월: 성능 최적화 및 모바일 앱 개발
- 11월: 베타 테스트 및 보안 강화
- 12월: 정식 런칭"""

    print("🔍 줄 단위 DIFF 분석")
    print("="*80)
    
    # 1. 줄 단위 diff (파이썬 기본 라이브러리)
    print("\n📋 줄 단위 변경사항:")
    print("-" * 50)
    
    original_lines = original.splitlines()
    modified_lines = modified.splitlines()
    
    diff = list(difflib.unified_diff(
        original_lines, 
        modified_lines, 
        lineterm='',
        n=0  # 컨텍스트 줄 수를 0으로 설정
    ))
    
    added_lines = []
    deleted_lines = []
    
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            added_lines.append(line[1:])  # + 기호 제거
        elif line.startswith('-') and not line.startswith('---'):
            deleted_lines.append(line[1:])  # - 기호 제거
    
    print(f"➕ 추가된 줄 ({len(added_lines)}개):")
    for i, line in enumerate(added_lines, 1):
        print(f"  {i}. {repr(line)}")
    
    print(f"\n🗑️ 삭제된 줄 ({len(deleted_lines)}개):")
    for i, line in enumerate(deleted_lines, 1):
        print(f"  {i}. {repr(line)}")

def test_char_diff():
    """문자 단위 diff 분석 (현재 방식)"""
    
    original = "서버 응답속도 지연 (평균 2.5초)"
    modified = "서버 응답속도 개선됨 (평균 1.2초) ← 성능 향상"
    
    print(f"\n🔍 문자 단위 DIFF 분석 (현재 방식)")
    print("="*50)
    
    print(f"원본: {repr(original)}")
    print(f"수정: {repr(modified)}")
    
    dmp = diff_match_patch()
    diffs = dmp.diff_main(original, modified)
    dmp.diff_cleanupSemantic(diffs)
    
    print(f"\nDIFF 청크 ({len(diffs)}개):")
    insertions = []
    deletions = []
    
    for i, (op, text) in enumerate(diffs):
        op_name = {-1: "🗑️삭제", 0: "⚪유지", 1: "➕추가"}[op]
        print(f"  [{i}] {op_name}: {repr(text)}")
        
        if op == 1:
            insertions.append(text)
        elif op == -1:
            deletions.append(text)
    
    print(f"\n결과:")
    print(f"추가: {repr(''.join(insertions))}")
    print(f"삭제: {repr(''.join(deletions))}")

if __name__ == "__main__":
    test_line_diff()
    test_char_diff() 