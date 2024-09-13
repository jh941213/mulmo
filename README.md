# 🍽️ AI 기반 비디오 분석 시스템

## 📚 개요
이 프로젝트는 사용자가 업로드한 비디오에서 **특정 이벤트**(예: 음식 섭취 시간 및 메뉴)를 자동으로 분석하여 정보를 제공하는 시스템입니다. **FastAPI**를 이용해 RESTful API를 제공하며, **OpenAI GPT-4 모델**, **LangChain**, **OpenCV**, **Google Generative AI** 등을 활용해 비디오를 분석합니다. 분석된 정보는 **벡터 데이터베이스**에 저장되어 빠르게 검색될 수 있습니다.

## 🎯 기능

- **📹 비디오 업로드 및 처리:** 사용자가 비디오 파일을 업로드하면 OpenCV를 통해 특정 구간을 분석하고 추출합니다.
- **🧠 GPT-4를 통한 분석:** 비디오에서 감지된 이벤트를 바탕으로 GPT-4가 자연어 분석을 통해 음식 섭취 시간 및 메뉴를 추출합니다.
- **🦜 LangChain 및 벡터 데이터베이스 연동:** 분석된 결과를 **Milvus** 벡터 데이터베이스에 저장하여, 빠르고 효율적으로 검색할 수 있습니다.
- **📝 QA 시스템 구축:** 사용자가 질문을 하면, 관련된 비디오 정보를 찾아주는 QA 시스템을 제공합니다.
- **🍽️ 동영상 클립 제공:** 분석된 특정 시간 구간의 동영상을 추출하여 사용자가 바로 확인할 수 있도록 제공합니다.

## ⚙️ 기술 스택

- **FastAPI:** RESTful API 서버
- **OpenAI GPT-4:** 자연어 처리 및 분석
- **LangChain:** 데이터 처리 및 분석 응답 관리
- **OpenCV:** 비디오 처리 및 클립 추출
- **Google Generative AI:** 비디오 파일 처리 및 AI 모델 연동
- **Chroma:** 벡터 데이터베이스로, 대규모 데이터를 효율적으로 검색할 수 있는 기술
- **Docker:** 컨테이너 기반 배포 및 실행 환경 관리

## 🚀 API 엔드포인트

### 1. `/upload-video` [POST]
- **설명:** 사용자가 비디오 파일을 업로드합니다.
- **입력:** `file` (UploadFile) - 업로드할 비디오 파일.
- **응답:** 업로드 성공 시 파일 경로와 상태를 반환합니다.

### 2. `/chat` [POST]
- **설명:** 사용자의 질문에 대해 비디오 관련 정보를 반환합니다. 비디오가 업로드된 경우, 해당 비디오에서 관련 구간을 분석하여 클립을 제공합니다.
- **입력:** `message` (String) - 사용자의 질문.
- **응답:** 비디오 경로, 분석된 음식 이름, 그리고 분석된 정보.

### 3. `/delete-video` [DELETE]
- **설명:** 서버에 업로드된 비디오를 삭제합니다.
- **입력:** `fileUrl` (String) - 삭제할 비디오 파일 경로.
- **응답:** 삭제 성공 메시지.

## 🛠️ 설치 및 실행

1. 저장소 클론:
    ```bash
    git clone https://github.com/jh941213/mulmo
    cd project-name
    ```

2. 의존성 설치:
    ```bash
    pip install -r requirements.txt
    ```

3. `.env` 파일 생성 및 API 키 설정:
    ```env
    OPENAI_API_KEY=your-openai-api-key
    GOOGLE_API_KEY=your-google-api-key
    ```

4. 서버 실행:
    ```bash
    uvicorn main:app --reload
    ```

5. API 문서는 `/docs`에서 Swagger UI로 확인 가능합니다.

## 🤝 기여

기여를 환영합니다! 이 프로젝트에 기여하려면 [기여 가이드](CONTRIBUTING.md)를 참고해 주세요.

---

**License:** MIT License
