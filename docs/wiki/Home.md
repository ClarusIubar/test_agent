# test_agent Wiki

## 목적
이 문서는 실제 사용자가 챗봇을 바로 실행할 수 있도록 단일 진입점, 인자 전달 방식, 실행 예시를 정리합니다.

## 단일 실행 진입점
- 엔트리 파일: [graph.py](../../graph.py)
- 근거: [graph.py](../../graph.py#L74) 의 __main__ 블록에서 CLI 인자를 받아 실행

## 실행 명령
- python graph.py "질문"

예시:
- python graph.py "삶이란 무엇일까?"
- python graph.py "search: 최신 LangGraph 특징 알려줘"

## 인자 전달 방법
- 질문 인자는 필수입니다.
- 인자가 없으면 Usage 메시지와 함께 종료됩니다.
- 공백이 있는 질문은 큰따옴표로 전달합니다.

## 동작 흐름
1. [graph.py](../../graph.py) 가 질문 문자열을 InputState로 변환
2. [agent_graph/core/graph_definition.py](../../agent_graph/core/graph_definition.py) 에서 그래프 실행
3. [chatbot.py](../../chatbot.py) 가 답변 생성 및 필요 시 tool_calls 구성
4. 최종 answer 문자열 출력

## search: 접두사 규칙
- 질문이 search: 로 시작하면 Tavily 검색 도구 경로를 사용합니다.
- 구현 위치: [chatbot.py](../../chatbot.py#L24)

## 환경 변수
- OpenAI 키: OPENAI_API_KEY
- Tavily 키: TAVILY_API_KEY
- 예시 파일: [ .env.example ](../../.env.example)

## 루트 파일 설명
- [graph.py](../../graph.py): 실행 진입점
- [chatbot.py](../../chatbot.py): 챗봇 노드
- [state.py](../../state.py): 상태 스키마
- [llm.py](../../llm.py): 모델 초기화
- [llm_smoke_test.py](../../llm_smoke_test.py): LLM 스모크 테스트

## 검증
- D:/miniconda3/envs/langgraph/python.exe -m unittest discover -s tests -v
