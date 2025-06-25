import json
import logging
import difflib
from typing import Dict, Any, List

logger = logging.getLogger("diff_util")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def extract_changes(original: str, modified: str) -> Dict[str, Any]:
    """
    원본과 수정본을 줄 단위로 비교하여 실제 추가/삭제된 내용만 정확히 추출합니다.
    
    Returns:
        {
            'insertions': List[str] - 실제 추가된 줄들
            'deletions': List[str] - 실제 삭제된 줄들  
            'has_changes': bool - 변경사항 존재 여부
        }
    """
    if not original and not modified:
        return {'insertions': [], 'deletions': [], 'has_changes': False}
    
    # 줄 단위로 분할
    original_lines = (original or '').splitlines()
    modified_lines = (modified or '').splitlines()
    
    # difflib로 줄 단위 diff 생성
    diff = list(difflib.unified_diff(
        original_lines, 
        modified_lines, 
        lineterm='',
        n=0  # 컨텍스트 줄 수를 0으로 설정하여 변경된 줄만 추출
    ))
    
    insertions = []
    deletions = []
    
    # diff 결과에서 추가/삭제 줄 추출
    for line in diff:
        if line.startswith('+') and not line.startswith('+++'):
            insertions.append(line[1:])  # + 기호 제거
        elif line.startswith('-') and not line.startswith('---'):
            deletions.append(line[1:])  # - 기호 제거
    
    # 변경사항 출력 (줄 단위)
    if deletions:
        print(f"🗑️ 삭제된 줄: {len(deletions)}개")
        for i, deletion in enumerate(deletions, 1):
            if deletion.strip():
                print(f"  [{i}] {deletion}")
    
    if insertions:
        print(f"➕ 추가된 줄: {len(insertions)}개")
        for i, insertion in enumerate(insertions, 1):
            if insertion.strip():
                print(f"  [{i}] {insertion}")
    
    return {
        'insertions': insertions,
        'deletions': deletions,
        'has_changes': bool(insertions or deletions)
    }


def compare_report_changes(draft_json: str, output_json: str) -> Dict[str, Any]:
    """
    Draft와 Output JSON의 report 내용을 키별로 비교하여 변경사항만 추출합니다.
    Draft의 reports에서 form_key를 추출하여, Output의 큰틀키 안에서 같은 form_key를 찾아 비교합니다.
    """
    # 1. Draft에서 reports 추출
    draft_reports = _extract_draft_contents(draft_json)
    
    if not draft_reports:
        print("⚠️ Draft에서 reports 없음")
        return {'unified_diff': '', 'comparisons': []}
    
    # 2. Draft의 form_keys를 사용해서 Output에서 해당 값들 추출
    draft_form_keys = set(draft_reports.keys())
    output_reports = _extract_output_contents(output_json, draft_form_keys)
    
    # 3. form_key별로 비교
    all_keys = draft_form_keys | set(output_reports.keys())
    comparisons = []
    unified_diffs = []
    
    print(f"📋 비교할 form_keys: {list(all_keys)}")
    
    for form_key in all_keys:
        draft_content = str(draft_reports.get(form_key, ""))
        output_content = str(output_reports.get(form_key, ""))
        
        if draft_content != output_content:
            print(f"\n🔍 [{form_key}] 변경사항 분석:")
            changes = extract_changes(draft_content, output_content)
            
            if changes['has_changes']:
                # 간단한 unified diff 생성
                diff_summary = _create_simple_diff(changes, form_key)
                unified_diffs.append(diff_summary)
                
                comparisons.append({
                    'key': form_key,
                    'draft_content': draft_content,
                    'output_content': output_content,
                    'changes': changes,
                    'diff_summary': diff_summary
                })
    
    return {
        'unified_diff': '\n\n'.join(unified_diffs),
        'comparisons': comparisons
    }


def _extract_draft_contents(json_data: str) -> Dict[str, Any]:
    """Draft JSON에서 report 구조의 내용을 추출합니다."""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        # Draft에서는 'report' 또는 'reports' 키에서 추출
        for key in ['reports']:
            if key in data and isinstance(data[key], dict):
                return data[key]
        return {}
        
    except Exception as e:
        print(f"❌ Draft JSON 파싱 오류: {e}")
        return {}


def _extract_output_contents(json_data: str, target_form_keys: set) -> Dict[str, Any]:
    """Output JSON에서 target_form_keys에 해당하는 값들을 추출합니다."""
    try:
        if isinstance(json_data, str):
            data = json.loads(json_data)
        else:
            data = json_data
        
        result = {}
        
        # Output의 모든 큰틀키 안에서 target_form_keys 찾기
        if isinstance(data, dict):
            for main_key, main_value in data.items():
                if isinstance(main_value, dict):
                    # 큰틀키 안에서 target_form_keys와 일치하는 것들 찾기
                    for form_key in target_form_keys:
                        if form_key in main_value:
                            result[form_key] = main_value[form_key]
        return result
        
    except Exception as e:
        print(f"❌ Output JSON 파싱 오류: {e}")
        return {}


def _create_simple_diff(changes: Dict[str, Any], key: str) -> str:
    """변경사항을 간단한 diff 형태로 표현합니다."""
    lines = [f"=== {key} ==="]
    
    if changes['deletions']:
        lines.append("- 삭제된 줄:")
        for deletion in changes['deletions']:
            if deletion.strip():  # 빈 문자열 제외
                lines.append(f"  - {deletion}")
    
    if changes['insertions']:
        lines.append("+ 추가된 줄:")
        for insertion in changes['insertions']:
            if insertion.strip():  # 빈 문자열 제외
                lines.append(f"  + {insertion}")
    
    return '\n'.join(lines) 