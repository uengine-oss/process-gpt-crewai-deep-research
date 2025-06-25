#!/usr/bin/env python3
"""
시스템의 줄 단위 diff 함수 테스트
"""

from src.parallel.feedback.diff_util import extract_changes, compare_report_changes
import json

def test_system_diff():
    """시스템의 줄 단위 diff 함수 직접 테스트"""
    
    print("🔍 시스템 줄 단위 DIFF 테스트")
    print("="*80)
    
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

    print("📄 원본 내용:")
    print(original[:200] + "..." if len(original) > 200 else original)
    
    print("\n📝 수정된 내용:")
    print(modified[:200] + "..." if len(modified) > 200 else modified)
    
    print("\n🔍 시스템 diff 분석:")
    print("-" * 50)
    
    # 시스템의 extract_changes 함수 직접 호출
    result = extract_changes(original, modified)
    
    print(f"\n📊 결과 요약:")
    print(f"추가된 줄: {len(result['insertions'])}개")
    print(f"삭제된 줄: {len(result['deletions'])}개")
    print(f"변경사항 있음: {result['has_changes']}")

def test_json_comparison():
    """JSON 형태의 리포트 비교 테스트"""
    
    print(f"\n{'='*80}")
    print("🔍 JSON 리포트 비교 테스트 (실제 시스템 방식)")
    print("="*80)
    
    # Draft JSON (원본)
    draft_json = {
        "reports": {
            "project_summary": """# 프로젝트 요약
- API 개발: 80% 완료
- UI 구현: 60% 진행 중
- 테스트: 미시작""",
            "issue_report": """## 주요 이슈
1. 성능 문제 있음
2. 보안 검토 필요"""
        }
    }
    
    # Output JSON (수정본)
    output_json = {
        "final_reports": {
            "project_summary": """# 프로젝트 요약
- API 개발: 95% 완료 ← 목표 초과
- UI 구현: 85% 진행 중 ← 빠른 진행
- 테스트: 30% 시작 ← 신규 추가
- 배포 준비: 10% ← 추가 항목""",
            "issue_report": """## 주요 이슈
1. 성능 문제 해결됨 ← 개선
2. 보안 검토 완료 ← 해결
3. SSL 인증서 갱신 필요 ← 새 이슈"""
        }
    }
    
    print("📋 Draft 내용:")
    for key, content in draft_json["reports"].items():
        print(f"  [{key}]: {content[:50]}...")
    
    print("\n📋 Output 내용:")
    for key, content in output_json["final_reports"].items():
        print(f"  [{key}]: {content[:50]}...")
    
    print("\n🔍 시스템 JSON 비교:")
    print("-" * 50)
    
    # 시스템의 compare_report_changes 함수 직접 호출
    result = compare_report_changes(
        json.dumps(draft_json), 
        json.dumps(output_json)
    )
    
    print(f"\n📊 비교 결과:")
    print(f"비교된 항목: {len(result['comparisons'])}개")
    print(f"변경사항 있음: {bool(result['unified_diff'])}")

if __name__ == "__main__":
    test_system_diff()
    test_json_comparison() 