"""
chung_variable_mapping.py의 매핑 데이터를 CSV 파일로 변환
"""
import csv
from chung_variable_mapping import variable_mapping

def export_to_csv(output_file="chung_variable_mapping.csv"):
    """변수 매핑을 CSV 파일로 출력"""
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['카테고리', '항목명', 'PISA변수', '출처', '설명']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for category, items in variable_mapping.items():
            for item_name, item_info in items.items():
                var = item_info.get("변수", "")
                source = item_info.get("출처", "")
                desc = item_info.get("설명", "")
                
                # 변수가 리스트인 경우 세미콜론으로 구분하여 하나의 셀에 저장
                if isinstance(var, list):
                    var_str = "; ".join(var)
                else:
                    var_str = var
                
                writer.writerow({
                    '카테고리': category,
                    '항목명': item_name,
                    'PISA변수': var_str,
                    '출처': source,
                    '설명': desc
                })
    
    print(f"CSV 파일이 생성되었습니다: {output_file}")
    
    # 통계 출력
    total_items = sum(len(items) for items in variable_mapping.values())
    print(f"\n총 {len(variable_mapping)}개 카테고리, {total_items}개 항목")
    
    for category, items in variable_mapping.items():
        print(f"  - {category}: {len(items)}개 항목")

if __name__ == "__main__":
    export_to_csv()
