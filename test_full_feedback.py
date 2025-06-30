#!/usr/bin/env python3
"""
하드코딩된 보고서로 실제 LLM 피드백 생성 전체 과정 테스트
"""

import asyncio
import json
import os
from dotenv import load_dotenv

from src.parallel.feedback.diff_util import compare_report_changes
from src.parallel.feedback.agent_feedback_analyzer import AgentFeedbackAnalyzer
from src.parallel.agents_repository import AgentsRepository

# 환경변수 로드
load_dotenv()

def get_hardcoded_reports():
    """하드코딩된 원본/수정된 보고서 데이터 (하나의 보고서만)"""
    
    # 원본 보고서 (Draft)
    draft_report = {
        "reports": {
            "project_status": """# 2024년 4분기 AI 플랫폼 개발 프로젝트

## 프로젝트 개요
- 프로젝트명: 지능형 고객 서비스 AI 플랫폼
- 진행 기간: 2024.10 ~ 2024.12
- 팀 구성: 개발 5명, 기획 2명, 디자인 1명

## 현재 진행 상황
### 백엔드 개발
- API 서버 구축: 85% 완료 ← 예상보다 빠른 진행
- 데이터베이스 설계: 완료
- AI 모델 통합: 75% 진행 중 ← 집중 투입으로 가속화

### 프론트엔드 개발  
- UI/UX 설계: 80% 완료
- 컴포넌트 개발: 60% 진행 중
- 반응형 웹 구현: 미착수

### AI 모델
- 자연어 처리 모델: 학습 완료 및 최적화 적용
- 감정 분석 모델: 정확도 87% 달성으로 테스트 통과 ← 목표 초과 달성
- 챗봇 대화 모델: 베타 버전 개발중

## 주요 이슈
1. AI 모델 정확도가 기대치 85% 대비 현재 78%
2. 데이터 전처리 과정에서 성능 병목 발생
3. 프론트엔드 일정 지연 (디자이너 휴가)
4. 클라우드 비용이 예산 초과 우려

## 다음 주 계획
- AI 모델 정확도 개선 작업
- 성능 최적화 진행
- 프론트엔드 추가 인력 투입 검토"""
        }
    }
    
    # 수정된 보고서 (Output)
    output_report = {
        "final_reports": {
            "project_status": """# 2024년 4분기 AI 플랫폼 개발 프로젝트

## 프로젝트 개요
- 프로젝트명: 지능형 고객 서비스 AI 플랫폼
- 진행 기간: 2024.10 ~ 2024.12
- 팀 구성: 개발 5명, 기획 2명, 디자인 1명

## 현재 진행 상황
### 백엔드 개발
- API 서버 구축: 85% 완료 ← 예상보다 빠른 진행
- 데이터베이스 설계: 완료
- AI 모델 통합: 75% 진행 중 ← 집중 투입으로 가속화

### 프론트엔드 개발  
- UI/UX 설계: 95% 완료 ← 디자인 시스템 도입으로 효율성 증대
- 컴포넌트 개발: 70% 진행 중 ← 외주 업체 투입 효과
- 반응형 웹 구현: 30% 시작 ← 병렬 작업으로 일정 단축

### AI 모델
- 자연어 처리 모델: 학습 완료 및 최적화 적용
- 감정 분석 모델: 정확도 87% 달성으로 테스트 통과 ← 목표 초과 달성
- 챗봇 대화 모델: 베타 버전 완성 ← 예정보다 1주 앞당김

## 주요 이슈
1. ~~AI 모델 정확도가 기대치 85% 대비 현재 78%~~ → 87% 달성으로 해결
2. 데이터 전처리 과정 성능 20% 개선 완료 ← Redis 캐싱 도입
3. ~~프론트엔드 일정 지연 (디자이너 휴가)~~ → 외주 업체 투입으로 해결
4. 클라우드 비용 최적화로 5% 절감 달성 ← 불필요한 리소스 정리

## 새로운 성과
- AI 모델 반응 속도 30% 향상
- 사용자 만족도 테스트에서 4.2/5.0 점수 획득
- 보안 감사 통과 및 인증 완료

## 다음 주 계획
- 통합 테스트 및 성능 검증
- 베타 사용자 피드백 수집 및 반영
- 정식 런칭 준비 및 마케팅 자료 제작"""
        }
    }
    
    return draft_report, output_report

async def load_real_agents():
    """실제 AgentsRepository에서 에이전트 데이터 로드"""
    
    print(f"\n🤖 실제 에이전트 데이터 로드")
    print("-" * 50)
    
    try:
        agents_repo = AgentsRepository()
        agents = await agents_repo.get_all_agents()
        
        print(f"✅ 에이전트 {len(agents)}개 로드 완료")
        
        # 에이전트 목록 출력
        for i, agent in enumerate(agents, 1):
            name = agent.get('name', 'Unknown')
            role = agent.get('role', 'Unknown')
            goal = agent.get('goal', 'No goal')[:200] + "..." if len(agent.get('goal', '')) > 200 else agent.get('goal', 'No goal')
            
            print(f"  [{i}] {name} ({role})")
            print(f"      목표: {goal}")
        
        return agents
        
    except Exception as e:
        print(f"❌ 에이전트 로드 오류: {e}")
        print("   Supabase 연결을 확인해주세요.")
        return []

async def test_full_feedback_process():
    """전체 피드백 생성 과정 테스트"""
    
    print("🚀 전체 피드백 생성 과정 테스트")
    print("="*100)
    
    # 0. 실제 에이전트 데이터 로드
    agents = await load_real_agents()
    if not agents:
        print("❌ 에이전트 데이터를 로드할 수 없습니다. 테스트를 종료합니다.")
        return
    
    # 1. 하드코딩된 보고서 데이터 준비
    print("\n📋 1단계: 보고서 데이터 준비")
    print("-" * 50)
    
    draft_report, output_report = get_hardcoded_reports()
    
    print(f"✅ Draft 보고서: {len(draft_report['reports'])}개 섹션")
    for key in draft_report['reports'].keys():
        content_length = len(draft_report['reports'][key])
        print(f"   - {key}: {content_length}자")
    
    print(f"✅ Output 보고서: {len(output_report['final_reports'])}개 섹션")  
    for key in output_report['final_reports'].keys():
        content_length = len(output_report['final_reports'][key])
        print(f"   - {key}: {content_length}자")
    
    # 2. DIFF 분석
    print(f"\n🔍 2단계: DIFF 분석")
    print("-" * 50)
    
    diff_result = compare_report_changes(
        json.dumps(draft_report, ensure_ascii=False),
        json.dumps(output_report, ensure_ascii=False)
    )
    
    print(f"✅ DIFF 분석 완료")
    print(f"   - 비교된 섹션: {len(diff_result.get('comparisons', []))}개")
    print(f"   - 변경사항 있음: {bool(diff_result.get('unified_diff'))}")
    
    if diff_result.get('comparisons'):
        total_insertions = sum(len(c.get('changes', {}).get('insertions', [])) 
                              for c in diff_result['comparisons'])
        total_deletions = sum(len(c.get('changes', {}).get('deletions', [])) 
                             for c in diff_result['comparisons'])
        print(f"   - 총 추가된 줄: {total_insertions}개")
        print(f"   - 총 삭제된 줄: {total_deletions}개")
    
    # 3. 에이전트 피드백 생성 (실제 LLM 호출)
    print(f"\n🤖 3단계: AI 에이전트 피드백 생성 (LLM 호출)")
    print("-" * 50)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정해주세요.")
        return
    
    analyzer = AgentFeedbackAnalyzer()
    
    try:
        print("🔄 LLM 분석 중...")
        
        # 원본 내용을 실제 텍스트 형태로 추출
        original_text = ""
        for key, content in draft_report['reports'].items():
            original_text += f"\n=== {key} ===\n{content}\n"
        
        feedback_list = await analyzer.generate_feedback_from_diff_result(
            diff_result=diff_result,
            original_content=original_text,  # JSON이 아닌 실제 텍스트로 전달
            todo_id="test_001",
            proc_inst_id="test_proc_001"
        )
        
        print(f"✅ 피드백 생성 완료: {len(feedback_list)}개")
        
        # 4. 결과 출력
        print(f"\n📊 4단계: 최종 결과")
        print("-" * 50)
        
        if feedback_list:
            print("🎯 생성된 에이전트 피드백:")
            for i, feedback in enumerate(feedback_list, 1):
                agent = feedback.get('agent', 'Unknown')
                message = feedback.get('feedback', 'No feedback')
                print(f"\n  [{i}] 에이전트: {agent}")
                print(f"      피드백: {message}")
        else:
            print("⚠️ 생성된 피드백이 없습니다.")
            
    except Exception as e:
        print(f"❌ LLM 호출 오류: {e}")
        print("   API 키 확인 또는 네트워크 연결을 확인해주세요.")

if __name__ == "__main__":
    print("🔍 하드코딩된 보고서로 실제 LLM 피드백 생성 테스트")
    
    # 전체 과정 실행
    asyncio.run(test_full_feedback_process()) 