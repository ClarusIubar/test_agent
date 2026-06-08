# test_agent

LangGraph 기반 챗봇 예제입니다. 현재 실행 진입점은 하나이며, 질문 문자열을 인자로 받아 답변을 출력합니다.

## 핵심 요약
- 단일 실행 진입점: [graph.py](graph.py)
- 실행 방식: CLI 인자 전달 필수
- 출력: 표준 출력으로 answer 텍스트 1회 출력

## 빠른 시작
### 1) 환경 준비
- Python 3.11+ 권장
- 가상환경(예: langgraph) 활성화

### 2) 의존성 설치
- pip install -r requirements.txt

### 3) 환경 변수 설정
- [ .env.example ](.env.example) 를 참고해 [.env](.env) 생성
- OpenAI 키와 Tavily 키를 설정

예시:
- OPENAI_API_KEY=...
- TAVILY_API_KEY=...

참고: 현재 [ .env.example ](.env.example)에는 OPEN_API_KEY가 적혀 있습니다. 실사용 시에는 OPENAI_API_KEY를 함께 설정하는 것을 권장합니다.

## 실행 방법
### 단일 진입점 실행
- python graph.py "질문"

예시:
- python graph.py "삶이란 무엇일까?"
- python graph.py "search: LangGraph가 뭐야?"

## 인자 전달 규칙
- 질문 인자가 1개 이상 필요합니다.
- 인자가 없으면 [graph.py](graph.py#L76)에 정의된 Usage 메시지로 종료됩니다.
- 여러 단어는 공백 포함 문자열로 넘기면 내부에서 합쳐서 처리합니다.

## 검색 트리거 규칙
- 질문이 search: 로 시작하면 도구 호출 경로를 탑니다.
- 관련 구현:
  - [chatbot.py](chatbot.py#L24)
  - [graph.py](graph.py#L30)

## 루트 파일 역할
- [graph.py](graph.py): 실행 진입점 + 그래프 조립 + invoke/stream 래퍼
- [chatbot.py](chatbot.py): 챗봇 노드 로직 및 tool_calls 생성 규칙
- [state.py](state.py): Input/Output/Overall 상태 스키마
- [llm.py](llm.py): LLM 초기화
- [llm_smoke_test.py](llm_smoke_test.py): LLM 단독 스모크 테스트
- [scripts/tavily_registry_smoke.py](scripts/tavily_registry_smoke.py): Tavily 레지스트리 스모크 테스트

## 테스트
- D:/miniconda3/envs/langgraph/python.exe -m unittest discover -s tests -v

## 위키
- 위키 원문(저장소 내): [docs/wiki/Home.md](docs/wiki/Home.md)
- GitHub Wiki에는 위 파일 내용을 그대로 옮겨 게시하면 됩니다.
