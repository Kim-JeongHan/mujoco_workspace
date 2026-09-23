# 구현 방향
# collect / train / evaluate 진입점을 tyro로 구성하고 config.py의 설정을 읽는다.
# collect는 전문가와 환경을 만들어 에피소드를 저장하고, train은 policy와 trainer를 조립한다.
# evaluate는 체크포인트의 정책·정규화·환경 설정을 복원해 공통 평가기를 실행한다.
# 실행 디렉터리를 생성하고 최종 설정, seed, 데이터셋 식별자를 함께 기록한다.
# 학습 계산은 trainer에 맡기고 여기서는 설정 검증과 객체 연결을 담당한다.
