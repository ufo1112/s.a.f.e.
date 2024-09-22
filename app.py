import streamlit as st
import cv2
import tempfile
import time
import os
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 스타일 적용
st.set_page_config(
    page_title="YOLO 객체 감지",
    page_icon="🔍",
    layout="wide"
)

# 커스텀 CSS 적용
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("style.css")

# 상수 정의
MODEL_PATH = 'best.pt'
FONT = cv2.FONT_HERSHEY_SIMPLEX

# 감지할 안전 장비
SAFETY_ITEMS = {"Mask": False, "Safety Vest": False, "Hardhat": False}

# 미리 정의된 색상 (BGR 형식)
PREDEFINED_COLORS = [
    (0, 0, 255), (0, 255, 0), (255, 0, 0),
    # 필요에 따라 다른 색상을 추가하세요
]

# 클래스별 색상을 저장할 딕셔너리
class_colors = {}

def assign_color(class_name):
    """각 클래스에 고유한 색상을 할당합니다."""
    if class_name not in class_colors:
        color_index = len(class_colors) % len(PREDEFINED_COLORS)
        class_colors[class_name] = PREDEFINED_COLORS[color_index]
    return class_colors[class_name]

def draw_detections(frame, results, model):
    """프레임에 바운딩 박스와 라벨을 그립니다."""
    safety_items_detected = SAFETY_ITEMS.copy()
    for r in results:
        for c in r.boxes:
            class_id = int(c.cls)
            class_name = model.names[class_id]
            x1, y1, x2, y2 = map(int, c.xyxy[0])

            color = assign_color(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), FONT, 0.5, color, 2)

            if class_name in safety_items_detected:
                safety_items_detected[class_name] = True
    return safety_items_detected

def display_good_message(frame):
    """모든 안전 장비가 감지되었을 때 'Good' 메시지를 표시합니다."""
    text = "Good"
    font_scale = 5
    font_thickness = 7
    text_size = cv2.getTextSize(text, FONT, font_scale, font_thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), FONT, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)

def main():
    # 헤더 섹션
    st.markdown("<h1 style='text-align: center; color: white;'>🔍 YOLO 객체 감지 앱</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: white;'>업로드한 비디오에서 안전 장비를 감지합니다.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # 사이드바 설정
    st.sidebar.header("설정")
    confidence_threshold = st.sidebar.slider("신뢰도 임계값", 0.0, 1.0, 0.5)

    # 파일 업로드 섹션
    uploaded_file = st.file_uploader("비디오 파일 선택", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        # YOLO 모델 로드
        model = YOLO(MODEL_PATH)
        cap = cv2.VideoCapture(video_path)

        # 진행 상황 표시
        progress_bar = st.progress(0)
        frame_placeholder = st.empty()
        warning_placeholder = st.empty()

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            current_frame += 1

            if not ret:
                st.success("비디오 처리가 완료되었습니다.")
                break

            results = model(frame, verbose=False)
            safety_items_detected = draw_detections(frame, results, model)

            if all(safety_items_detected.values()):
                display_good_message(frame)
                warning_placeholder.empty()
            else:
                missing_items = [item for item, detected in safety_items_detected.items() if not detected]
                warning_message = f"⚠️ 감지되지 않은 안전 장비: {', '.join(missing_items)}"
                warning_placeholder.warning(warning_message)

            # 프레임을 RGB로 변환하여 표시
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

            # 진행 상황 업데이트
            progress = current_frame / frame_count
            progress_bar.progress(progress)

            # 프레임 속도를 조절하기 위한 딜레이
            time.sleep(0.01)

        cap.release()
        os.unlink(tfile.name)
    else:
        st.info("비디오 파일을 업로드해주세요.")

if __name__ == '__main__':
    main()
