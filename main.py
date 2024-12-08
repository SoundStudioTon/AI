from flask import Flask, request, jsonify
from PIL import Image
from pydub import AudioSegment
from scipy.io.wavfile import write

import io
import base64

import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import librosa
import random
import json


# # # 집중도 반환 코드
class Concentration:
    def __init__(self):
        # 집중도 예측 모델 불러오기
        self.model = tf.keras.models.load_model('concentration_predict_model.h5')

        # Mediapipe의 얼굴 탐지 객체, 그물망 객체 생성 및 초기화
        self.face_detection = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                                                         min_detection_confidence=0.5)

        # 집중도 라벨링
        self.label = {
            0: '얼굴 없음',
            1: '집중함',
            2: '집중하지 않음',
            3: '졸음'
        }

        # 랜드마크 좌표 / 68개
        self.landmarks = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300, 33, 160, 158, 133, 153, 144, 362, 385, 387,
                          263, 373, 380, 6, 197, 195, 5, 4, 239, 44, 1, 274, 459, 61, 40, 37, 0, 267, 270, 291, 91, 181,
                          84, 17, 314, 405, 321, 81, 38, 13, 268, 311, 310, 127, 234, 137, 215, 172, 136, 149, 148, 152,
                          377, 378, 365, 397, 435, 366, 454, 356]

    def detect_face(self, image):
        height, width, _ = image.shape
        face_crops = []

        # 이미지를 RGB로 변환
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 이미지에서 얼굴 탐지
        face_results = self.face_detection.process(image_rgb)
        mesh_results = self.face_mesh.process(image_rgb)

        # 탐지된 얼굴에 바운딩 박스, 랜드마크 그리기
        if face_results and face_results.detections:
            detection = face_results.detections[0]

            # 바운딩 박스 그리기
            bbox = detection.location_data.relative_bounding_box
            xmin, ymin = int(bbox.xmin * width), int(bbox.ymin * height)
            w, h = int(bbox.width * width), int(bbox.height * height)
            box = xmin, ymin, w, h
            cv2.rectangle(image_rgb, box, color=(0, 255, 0), thickness=5)

            # 랜드마크 그리기
            for face_landmarks in mesh_results.multi_face_landmarks:
                for idx in self.landmarks:
                    x, y = int(face_landmarks.landmark[idx].x * width), int(face_landmarks.landmark[idx].y * height)
                    cv2.circle(image_rgb, (x, y), radius=8, color=(0, 0, 255), thickness=-1)

            # 얼굴 부분 자르기
            face_crop = image_rgb[ymin:ymin + h, xmin:xmin + w]
            face_crops.append(face_crop)

        return face_crops

    def preprocess_face(self, face):
        input_size = (224, 224)
        face_resized = tf.image.resize(face, input_size)

        # 모델 입력으로 사용하기 위해 배치 차원 추가
        face_input = tf.expand_dims(face_resized, axis=0)

        return face_input

    def predict_concentration(self, face_input):
        # 예측 수행
        prediction = self.model.predict(face_input).argmax()

        # 집중함, 집중하지 않음, 졸음 3개로 축소 (흥미로운, 차분함 합침)
        if prediction <= 1:
            prediction = 1
        elif prediction <= 3:
            prediction = 2
        else:
            prediction = 3

        return self.label[prediction]


app = Flask(__name__)
concentration = Concentration()


@app.route('/predict', methods=['POST'])
def predict():
    # 클라이언트 이미지 받기
    if 'file' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = np.array(image)

    # 얼굴 탐지 및 집중도 예측
    face = concentration.detect_face(image)

    if face:  # 얼굴 탐지 O
        face_input = concentration.preprocess_face(face[0])
        prediction = concentration.predict_concentration(face_input)

        return jsonify({'prediction': prediction})
    else:  # 얼굴 탐지 X
        return jsonify({'prediction': 0})


# # # 소음 변형 코드
class NoiseAgent:
    def __init__(self, action_space):
        self.action_space = action_space  # 어떤 행동을 할지 저장한 리스트
        self.len_action_space = len(action_space)  # 행동의 가짓수

        self.value_table = np.zeros((100, self.len_action_space))  # 상태-가치 저장할 table

        self.alpha = 0.1  # 학습률
        self.gamma = 0.9  # 할인률
        self.epsilon = 0.3  # 선택 비율

    # 테이블 초기화 함수
    def set_value_table(self, methods_value):
        self.value_table = np.array(methods_value)

    # 다음 상태 인덱스 함수
    def get_next_idx(self, prev_idx, prev_action, reward):
        temp = self.value_table[prev_idx]
        temp[prev_action] += reward

        return int(sum(temp)) % 100

    # 행동 선택 함수
    def choose_action(self, state_idx):
        rand = round(random.random(), 2)  # 랜덤 숫자
        max_idx = np.argmax(self.value_table[state_idx])

        if rand <= self.epsilon:  # epsilon*100 % 만큼의 확률로 가장 가치가 큰 행동 실행
            return max_idx

        # (1-epsilon) * 100 %의 확률로 다른 행동 실행
        return random.choice([idx for idx in range(self.len_action_space) if idx != max_idx])

    # 행동의 보상으로 테이블 업데이트
    def update_value_table(self, action_idx, reward, state_idx, next_state_idx):
        best_action = np.argmax(self.value_table[next_state_idx])
        current_est = reward + self.gamma * self.value_table[next_state_idx, best_action]
        next_est = self.value_table[state_idx, action_idx]
        self.value_table[state_idx, action_idx] += self.alpha * (current_est - next_est)


# 소음 파일 불러오는 함수
def load_noise(file_path):
    noise, sr = librosa.load(file_path, sr=44100, mono=False)
    return noise, sr


# 특정 주파수 대역 강조 함수
def emphasize_frequency(audio, sr, target_freq, gain):
    # FFT를 이용해 신호를 주파수 도메인으로 변환
    fft_signal = np.fft.fft(audio)

    # 각 주파수 대역별로 실행
    for i in range(4):
        # 강조할 대역폭 설정
        if gain[i] >= 0.0:
            g_rate = 1.0 + 0.2 * gain[i]
        else:
            g_rate = 1.0 * 0.1 * gain[i]

        band_width = [
            (0, target_freq[i][0] - 1, 1.0),
            (target_freq[i][0] - 1, target_freq[i][1] + 1, g_rate),
            (target_freq[i][1] + 1, sr // 2, 1.0)
        ]

        # 변환 수행
        for band in band_width:
            start_freq, end_freq, g = band

            # 해당 대역의 인덱스
            start_idx = start_freq * len(fft_signal) // sr
            end_idx = end_freq * len(fft_signal) // sr

            # 해당 대역 변환 / 대칭 성분도 같이
            fft_signal[start_idx:end_idx + 1] *= g
            fft_signal[-end_idx - 1:-start_idx] *= g

    # 다시 시간 도메인으로 변환 후 반환
    emphasized_audio = np.fft.ifft(fft_signal).real

    return emphasized_audio


# 소음 변형 함수
def transform_noise(audio, sr, f_r_list, v_r):
    # 주파수 변형 / 최대 최소로 바뀔 수 있는 범위 설정
    for i, f in enumerate(f_r_list):
        if f < -5.0:
            f_r_list[i] = -5.0
        elif f > 10.0:
            f_r_list[i] = 10.0

    # 채널이 2개라 둘 다 진행
    emphasized_left = emphasize_frequency(audio[0], sr, [(20, 250), (250, 4000), (4000, 8000), (8000, 16000)], f_r_list)
    emphasized_right = emphasize_frequency(audio[1], sr, [(20, 250), (250, 4000), (4000, 8000), (8000, 16000)], f_r_list)
    audio = np.vstack([emphasized_left, emphasized_right])

    # 소리 크기 변형
    if v_r > 10.0:
        v_r = 10.0
    elif v_r < -5.0:
        v_r = -5.0

    audio = audio * (1.0 + v_r * 0.1)

    return audio


# 소음 변형 후 정보 넘기는 함수
@app.route('/transform', methods=['POST'])
def return_transformed_noise_data():
    # 데이터 받기
    data = request.get_json()

    # 데이터가 없을 경우
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    # 행동 종류 / 10가지
    action_space = ['freq_l_up', 'freq_l_down', 'freq_m_up', 'freq_m_down', 'freq_h_up',
                    'freq_h_down', 'freq_uh_up', 'freq_uh_down', 'volume_up', 'volume_down']

    # Agent 생성
    agent = NoiseAgent(action_space)

    # 테이블 초기화
    value_list = data['methods_value_list']
    value_list = json.loads(value_list)
    agent.set_value_table(value_list['methods_value_list'])

    # 전의 변형된 소음의 보상을 사용해 테이블 업데이트
    next_idx = agent.get_next_idx(int(data['prev_idx']), int(data['prev_method_idx']), int(data['reward']))
    agent.update_value_table(int(data['prev_method_idx']), int(data['reward']),
                             int(data['prev_idx']), next_idx)

    # 사용자별 소음 번호에 맞춰 소음 선택 / 1 - pink, 2 - wave, 3 - white
    noise_file = f'noise_{data["noise_number"]}.wav'
    noise, sr = load_noise(noise_file)

    # 행동 선택하기
    action_idx = agent.choose_action(next_idx)

    rate_list = [int(data['freq_1']), int(data['freq_2']), int(data['freq_3']), int(data['freq_4']), int(data['volume'])]
    if action_idx % 2 == 0:  # up
        rate_list[action_idx // 2] += 1
    else:  # down
        rate_list[action_idx // 2] -= 1

    # 소음 변형하기
    transformed_noise = transform_noise(noise, sr, rate_list[:4], rate_list[4])
    transformed_noise = transformed_noise.T
    transformed_noise_io = io.BytesIO()

    transformed_noise_int16 = (transformed_noise * 32767).astype(np.int16)
    write(transformed_noise_io, sr, transformed_noise_int16)
    transformed_noise_io.seek(0)

    # mp3 파일로 변형
    transformed_noise_wav = AudioSegment.from_file(transformed_noise_io, format='wav', frame_rate=sr)

    mp3_io = io.BytesIO()
    transformed_noise_wav.export(mp3_io, format='mp3')
    mp3_io.seek(0)

    # Base64 인코딩
    mp3_b64 = base64.b64encode(mp3_io.read()).decode('utf-8')

    # 값 반환
    return jsonify({
        'freq_1': rate_list[0], 'freq_2': rate_list[1], 'freq_3': rate_list[2], 'freq_4': rate_list[3],
        'volume': rate_list[4], 'prev_state_idx': next_idx,
        'methods_value_list': json.dumps({'methods_value_list': agent.value_table}),
        'prev_method_idx': int(action_idx),
        'transformed_noise': mp3_b64
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
