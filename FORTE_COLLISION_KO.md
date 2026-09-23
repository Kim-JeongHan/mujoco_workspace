# Forte collision 제작 과정

## 1. 대상과 변경 목적

대상은 사용자가 제공한 `ForteV1_RobStride.zip`에서 변환한 MuJoCo 모델이다. 원본 URDF에는 외형을 표시하는 `visual` 정보는 있지만 `collision` 정의는 없다. 따라서 시뮬레이션에서 접촉을 계산할 형상을 별도로 구성했다.

첫 변환에서는 베이스 원통 1개, 팔 capsule 6개, 손가락 패드 box 2개를 사용했다. 외형과 비교한 결과 베이스·어깨·손바닥·손가락 외측에 충돌 형상이 빠진 부분이 컸다. 이번 작업에서는 **같은 CAD 모델의 collision을 실제 부품 형상에 맞춰 보완**했다.

| 구성 | 첫 변환 | 보완 후 |
| --- | ---: | ---: |
| CAD 부품 mesh를 참조하는 collision geom | 0개 | 48개 |
| 원통 | 1개 | 0개 |
| Capsule | 6개 | 1개 |
| 손가락 접촉 패드 box | 2개 | 2개 |
| 합계 | 9개 | 51개 |

48개는 충돌용 `geom`의 수다. 새로운 mesh 파일을 48개 만든 것이 아니라,
이미 있는 CAD mesh를 충돌용 geom에서도 참조하도록 했다.

![같은 CAD와 같은 자세에서 비교한 collision](outputs/collision-update-20260922/comparison.png)

## 2. CAD에서 충돌에 사용할 부품 선택

[변환 스크립트](scripts/import_forte_robstride.py)의 `COLLISION_PARTS`에
각 body에서 충돌에 사용할 mesh 이름을 명시했다.

선택 기준은 외부 물체와 접촉할 수 있는 주요 구조물이다.
베이스 판, 어깨 지지대, 팔 프레임, 모터 외형, 손바닥 구조물, 손가락 판 등을 포함했다.
작은 나사와 내부 전자부품은 외형 표시용으로 유지했다.

| Body | 주요 선택 부품 |
| --- | --- |
| `base_link` | 베이스 판, 베이스 모터, 구동 드럼 |
| `main_drum` | 회전 드럼, 좌우 지지대, 어깨 모터와 외측 부품 |
| `lefthinge` | 힌지, shoulder roll 구조물, 모터, 덮개 |
| `upperarmright` | 위팔 좌우 프레임, 모터, 덮개와 외측 부품 |
| `elbowlink` | 팔꿈치 프레임과 모터 |
| `spur_gear__40_teeth_` | 아래팔 말단 구조물과 기어; 직선 튜브는 capsule 사용 |
| `spiral_gear_2` | 손목 좌우 구조물, 모터, 기어 |
| `part_8_2` | 손바닥과 그리퍼 본체의 주요 구조물 |
| `part_1_35`, `part_1_40` | 양쪽 손가락 판과 연결 부품 |

부품 선택은 이 CAD에 맞춰 명시한 목록이다. 임의의 로봇에 자동으로 최적 collision을

## 3. Visual geom을 collision geom으로 연결

`add_collisions()`는 선택한 부품의 visual geom을 복사한 뒤,
같은 body에 충돌용 geom으로 추가한다.

기존의 `mesh`, 위치 `pos`, 방향 `quat`를 재사용하므로 부품의 CAD 좌표계를 유지한다.
베이스와 손가락에는 뒤에서 설명하는 작은 collision 위치 보정만 적용한다.

핵심 처리는 다음과 같다.

```python
geom = deepcopy(visual)
geom.attrib.update(
    name=f"{name}_collision_{index}",
    contype="1",
    conaffinity="1",
    group="3",
    rgba="0.4 0.4 0.45 0.3",
)
body.append(geom)
```

| 속성 | 외형용 geom | 충돌용 geom |
| --- | --- | --- |
| `mesh` | 원본 CAD mesh 참조 | 동일한 mesh 참조 |
| `contype`, `conaffinity` | 둘 다 `0` | 둘 다 `1` |
| `group` | `1` | `3` |
| 역할 | 외형 표시 | 충돌 계산과 디버깅 표시 |

`group`은 표시 그룹이다. 실제 충돌 참여 여부는 `contype`, `conaffinity`와
접촉 제외 설정 등에 의해 결정된다. 화면에서 group 3을 숨겨도 충돌 계산은 유지된다.

복사한 geom의 `density=0`을 유지하고, body에 명시된 질량과 관성을 사용한다.
충돌용 geom을 추가했다고 로봇의 질량을 중복해서 더하지 않는다.

### Convex hull을 사용하는 의미

MuJoCo의 일반적인 mesh 충돌 계산은 mesh의 **convex hull, 즉 볼록 외곽**을 사용한다. 선택한 CAD 부품 각각을 별도 geom으로 배치해 충돌을 계산하도록 했다. 별도의 mesh 파일 생성이나 자동 convex decomposition 단계는 추가하지 않았다.

이 방식은 부품 외곽을 반영하지만, 부품 내부의 구멍이나 오목한 홈까지 정확하게 보존하지는 않는다. 특히 오목한 구조물의 빈 공간은 hull 안에 포함될 수 있다. 그래서 외형과의 비교뿐 아니라 동작 중 불필요한 자기 충돌이 생기는지도 확인했다.

## 4. 아래팔 튜브는 치수에 맞춘 capsule 사용

아래팔의 `Part_57`은 길이 약 285.5mm, 외측 반지름 약 18.5mm인 직선 튜브다.
이 부분은 단순한 형태이므로 반지름 19mm의 capsule을 사용했다.

```xml
<geom name="forearm_collision"
      type="capsule"
      fromto="0 0 -0.019 0 0 -0.2665"
      size="0.019"
      group="3"/>
```

`fromto`는 capsule 양 끝 구의 중심이다. 중심 사이 거리 247.5mm에 양 끝 반지름 38mm를 더하면 전체 길이는 285.5mm가 된다. 튜브의 말단 구조물과 기어는 별도의 CAD 부품 collision으로 처리한다.

## 5. 베이스와 손가락의 작은 위치 보정

### 베이스: 로컬 z 방향으로 0.1mm 이동

베이스 collision만 로컬 z 방향으로 0.1mm 올렸다. 장착면과 정확히 맞닿은 위치에서 좌표 반올림 때문에 미세한 초기 접촉이 발생하는 것을 피하기 위한 간격이다. 원본 visual과 관절 좌표는 이동하지 않았다.

### 손가락: 외측 collision을 바깥쪽으로 0.5mm 이동

손가락에는 외측 CAD collision과 안쪽 접촉 패드를 함께 사용한다.외측 collision을 각 손가락의 바깥 방향으로 0.5mm 이동시켜,물체가 안쪽 패드와 먼저 접촉하게 했다.

패드 자체의 위치를 옮긴 것이 아니라 외측 collision의 위치를 보정한 것이다. 따라서 패드는 외측 collision에 비해 안쪽으로 0.5mm 돌출된 관계가 된다.

## 6. 손가락 접촉 패드

안쪽 패드는 CAD 손가락의 평평한 접촉면에 맞춘 box다.
`size="0.002 0.015 0.01"`은 반길이이므로 실제 크기는 4 × 30 × 20mm다.

패드에는 다음 접촉 속성을 적용했다.

| 설정 | 값 | 적용 목적 |
| --- | --- | --- |
| `friction` | `1 0.02 0.001` | 기존 마찰 계수 유지 |
| `condim` | `4` | 미끄럼 마찰과 접촉 법선축 주위의 회전 마찰 사용 |
| `priority` | `1` | 기본 우선순위 물체와 접촉할 때 패드 설정 우선 적용 |
| `solref` | `0.006 1` | 접촉 응답의 시간상수와 감쇠 설정 |
| `solimp` | `0.995 0.999 0.001` | 접촉 제약의 강도를 조정해 과도한 밀림 감소 |

이 값들은 시뮬레이션용 접촉 설정이다. 실제 패드 재질을 측정해 얻은 물성값은 아니다.
전체 장면의 solver 옵션은 바꾸지 않고 패드 geom에 설정했다.

## 7. 외형과 collision을 검증한 방법

### 표면과 충돌 형상 사이 거리 측정

같은 CAD 모델의 같은 초기 자세에서 여섯 방향으로 ray를 쏘아, 각 ray가 처음 만나는 보이는 표면을 표본으로 선택했다. 총 2,034개 표면 지점을 보완 전후에 동일하게 사용했다.

각 표면 지점에 반지름 10µm의 작은 구를 놓고, `mj_geomDistance`로 로봇의 collision geom들과의 최소 거리를 계산했다. 측정값에는 구의 반지름을 보정했다. 단순히 화면상 mesh가 겹치는지 보는 대신 MuJoCo의 충돌 거리 계산을 사용한 것이다.

| 부위 | 보완 전 최대 표본 간격 | 보완 후 최대 표본 간격 |
| --- | ---: | ---: |
| 베이스 | 119.3mm | 1.4mm |
| 어깨 드럼·지지대 | 81.0mm | 3.6mm |
| 손바닥·그리퍼 본체 | 61.3mm | 2.4mm |
| 왼쪽 손가락 | 27.7mm | 0.5mm |
| 오른쪽 손가락 | 33.5mm | 0.5mm |

표면에서 collision까지 5mm 넘게 떨어진 표본은 **1,215개에서 0개로 줄었다.** 이는 조사한 표면 지점에서 충돌 누락이 줄었다는 결과다. 또한 coverage가 좋아져도 오목한 공간을 과하게 채우는 문제는 별도로 살펴봐야 한다.

### 자기 충돌 검사

PD 예제 경로의 관절 자세를 보간하고, 손가락을 열거나 닫은 경우를 포함해 168개 자세를 검사했다. 이 범위에서는 0.1mm보다 깊은 자기 관통이 발견되지 않았다.
기존 인접 팔 body 사이의 접촉 제외 설정은 유지했다.

이미 발견했던 베이스·어깨·손바닥·손가락의 누락 지점도 [회귀 테스트](tests/test_forte_asset.py)에 추가했다.

## 8. UI에서 확인하는 방법

프로젝트 루트에서 실행한다.

```bash
uv run python examples/manipulator.py --robot forte --controller pd
```

1. 왼쪽 `Group enable → Geom groups`를 연다.
2. `Geom 3`을 켜면 collision이 표시된다. 숫자 `3`으로도 전환할 수 있다.
3. `Geom 1`을 끄면 CAD visual이 숨겨져 collision만 보기 쉽다.
4. `H`로 mesh의 convex hull 표시를 전환하면 충돌 계산에 사용하는 외곽을 볼 수 있다.

## 9. 관련 파일

| 파일 | 내용 |
| --- | --- |
| [변환 스크립트](scripts/import_forte_robstride.py) | `COLLISION_PARTS`, `add_collisions()`, capsule 및 패드 생성 |
| [MuJoCo 모델](src/mujoco_lab/assets/robot/forte/robot.xml) | 생성된 collision 정의 |
| [원본 CAD 스냅샷](third_party/fortev1-robstride/) | 변경하지 않고 보존한 원본 파일 |
| [충돌 회귀 테스트](tests/test_forte_asset.py) | 누락 표면 지점과 동작 경로의 자기 충돌 검사 |
| [표면 거리 측정 결과](outputs/collision-update-20260922/surface-distances.json) | 보완 후의 표본별 거리와 부위별 집계 |
| [보완 전후 비교 이미지](outputs/collision-update-20260922/comparison.png) | 같은 모델·같은 자세에서의 형상 비교 |

이미지와 측정 JSON은 현재 workspace의 `outputs/`에 저장된 로컬 검증 산출물이다. collision을 재생성하는 작업은 변환 스크립트에 포함되어 있으며, 원본 CAD mesh 파일 자체는 수정하지 않는다.
