# Base Image 선택
FROM python:3.9-slim

# 필수 패키지 설치
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y ffmpeg

# 작업 디렉토리 설정
WORKDIR /soundstudio

# 필요한 파일 복사
COPY main.py .
COPY requirements.txt .
COPY concentration_predict_model.h5 .
COPY noise_files .

# 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# Flask 서버 실행
CMD ["python", "main.py"]