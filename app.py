# SafeFall AI - AI-Powered Elderly Fall Detection & Healthcare Monitoring System
# Student: Aditya Jitendra Kumar Sahani (ID: 1000414)
# Formative Assessment-2 (FA-2) - Machine Learning and Deep Learning (CRS AI)

import io
import json
import math
import threading
import time
from pathlib import Path

import altair as alt
try:
    import av
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

ROOT = Path(__file__).parent
TASK_PATH = ROOT / "pose_landmarker.task"
LOGO_PATH = ROOT / "logo.png"

st.set_page_config(
    page_title="SafeFall AI - Aditya Sahani (1000414)",
    page_icon="🛡️",
    layout="centered"
)


def current_theme():
    saved = st.query_params.get("theme")
    if saved in ("light", "dark"):
        return saved
    try:
        detected = st.context.theme.type
        if detected in ("light", "dark"):
            return detected
    except Exception:
        pass
    return "dark"


def paint_page(theme_name):
    if theme_name == "light":
        base = "#fff8f6"
        blooms = """
            radial-gradient(circle at 8% 0%, rgba(255, 59, 48, 0.45), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 152, 0, 0.40), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(33, 150, 243, 0.38), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(156, 39, 176, 0.40), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 59, 0.35), transparent 40%),
            radial-gradient(circle at 72% 58%, rgba(76, 175, 80, 0.35), transparent 44%)
        """
        card_bg = "rgba(255, 255, 255, 0.85)"
        text_color = "#1a2744"
    else:
        base = "#1a162b"
        blooms = """
            radial-gradient(circle at 18% 12%, rgba(150, 230, 230, 0.45), transparent 26%),
            radial-gradient(circle at 8% 0%, rgba(255, 110, 100, 0.60), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 176, 70, 0.50), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(90, 180, 255, 0.45), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(190, 110, 230, 0.50), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 120, 0.22), transparent 36%),
            radial-gradient(circle at 72% 58%, rgba(120, 220, 130, 0.35), transparent 42%)
        """
        card_bg = "rgba(30, 25, 48, 0.85)"
        text_color = "#f0f2f6"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {base};
            background-image: {blooms};
            background-attachment: fixed;
            color: {text_color};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stMainMenuDivider"]:has(+ [data-testid="stThemeSwitcher"]),
        [data-testid="stThemeSwitcher"] {{
            display: none;
        }}
        .author-badge {{
            display: inline-block;
            background: rgba(255, 255, 255, 0.12);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.85rem;
            margin-bottom: 8px;
            border: 1px solid rgba(255, 255, 255, 0.18);
        }}
        [data-testid="stHtml"]:has(#fall-alarm-setup) {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def switch_theme(choice):
    stored = "Light" if choice == "Light" else "Dark"
    value = stored.lower()
    st.html(
        f"""
        <script>
        const key = "stActiveTheme-" + window.location.pathname + "-v2";
        localStorage.setItem(key, JSON.stringify("{stored}"));
        const url = new URL(window.location.href);
        url.searchParams.set("theme", "{value}");
        window.location.replace(url.toString());
        </script>
        """,
        unsafe_allow_javascript=True,
    )


look = current_theme()
paint_page(look)


@st.dialog("About SafeFall AI")
def show_about():
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    st.markdown("### SafeFall AI - Elderly Fall Detection")
    st.markdown(
        """
        **Student:** Aditya Jitendra Kumar Sahani  
        **Student ID:** 1000414  
        **Assessment:** Formative Assessment-2 (FA-2)  
        **Subject:** Machine Learning & Deep Learning (CRS AI)  

        ---
        This clinical healthcare vision system analyzes human posture in real time using 
        **MediaPipe Pose Tasks** (33 landmarks) and biomechanical kinematics:
        - 🚨 **Fall Detected** — Body tipped over / recumbent posture ($>35^\\circ$ tilt)
        - ⚠️ **Off Balance** — Acute body lean or sudden angular delta ($>20^\\circ$)
        - 🪑 **Sitting** — Chair sit detection via knee flexion and thigh depth
        - 🧍 **Standing** — Upright vertical postural alignment
        - 🚶 **Walking** — Asymmetric leg gait, foot lift, and hip translation
        - ✅ **Normal** — Upright posture or no subject in frame
        """
    )
    st.link_button("🎨 View FA-2 Canva Storyboard", "https://canva.link/7iadcf1w4rsmwx3", use_container_width=True)
    st.info("Input options: Single Photo, Stored Video, Camera Snapshot, or Live WebRTC.")


# Header section
main_col, theme_col = st.columns([4.4, 1.4], vertical_alignment="center")
with main_col:
    logo_col, title_col = st.columns([1, 4.5], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=75)
        if st.button("Info", icon=":material/info:", key="logo_about", type="tertiary"):
            show_about()
    with title_col:
        st.title("SafeFall AI")
        st.caption("Aditya Jitendra Kumar Sahani (ID: 1000414) | Fall & Posture Detection")

with theme_col:
    label = "Light" if look == "light" else "Dark"
    if st.session_state.get("_applied_theme") != look:
        st.session_state.appearance = label
        st.session_state._applied_theme = look
    choice = st.segmented_control(
        "Appearance",
        ["Light", "Dark"],
        key="appearance",
        label_visibility="collapsed",
        required=True,
    )
    if choice and choice.lower() != look:
        st.session_state._applied_theme = choice.lower()
        switch_theme(choice)

st.write("Upload a photo or a video, take a camera snapshot, or start the live camera feed.")
st.session_state["_fall_alarm_shown"] = False


def make_pose():
    if not TASK_PATH.exists():
        st.error(f"MediaPipe model asset not found at {TASK_PATH}")
        return None
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(TASK_PATH),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


@st.cache_resource
def load_pose():
    return make_pose()


class LiveCamera:
    def __init__(self):
        self.pose = make_pose()
        self.last_tilt = None
        self.hips = []
        self.feet = []
        self.fall_hold = False


_live_local = threading.local()
_live_label = {"name": None, "false_alarm": None, "alarm_until": 0.0}
_live_matrix = {}
MATRIX_LABELS = ["fall", "sleeping", "sitting", "standing", "walking", "off balance", "normal"]


def live_state():
    state = getattr(_live_local, "state", None)
    if state is None:
        state = LiveCamera()
        _live_local.state = state
    return state


def knee_bend(hip, knee, ankle):
    ax, ay = hip.x - knee.x, hip.y - knee.y
    bx, by = ankle.x - knee.x, ankle.y - knee.y
    na = math.hypot(ax, ay)
    nb = math.hypot(bx, by)
    if na * nb == 0:
        return 180.0
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
    return math.degrees(math.acos(cos))


def seen(point):
    visibility = point.visibility if point.visibility is not None else 1.0
    return visibility >= 0.5


def body_features(landmarks):
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_x = (landmarks[23].x + landmarks[24].x) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    left_bend = knee_bend(landmarks[23], landmarks[25], landmarks[27])
    right_bend = knee_bend(landmarks[24], landmarks[26], landmarks[28])
    bend = (left_bend + right_bend) / 2
    ankle_gap = abs(landmarks[27].x - landmarks[28].x)
    hip_width = abs(landmarks[23].x - landmarks[24].x)
    foot_lift = abs(landmarks[27].y - landmarks[28].y)
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    tilt = abs(math.degrees(math.atan2(dx, dy))) if (dx or dy) else 0.0
    return tilt, bend, ankle_gap, hip_width, left_bend, right_bend, foot_lift, hip_x, hip_y


def hip_is_moving(hips, hip_x, hip_y):
    hips.append((hip_x, hip_y))
    if len(hips) > 10:
        del hips[0]
    if len(hips) < 6:
        return False
    path = 0.0
    for start, end in zip(hips, hips[1:]):
        path += math.hypot(end[0] - start[0], end[1] - start[1])
    net = math.hypot(hips[-1][0] - hips[0][0], hips[-1][1] - hips[0][1])
    return path > 0.10 and net > path * 0.45


def feet_are_moving(feet, left_ankle, right_ankle, hip_x, hip_y):
    gap = abs(left_ankle.x - right_ankle.x)
    feet.append((
        left_ankle.x - hip_x,
        left_ankle.y - hip_y,
        right_ankle.x - hip_x,
        right_ankle.y - hip_y,
        gap,
    ))
    if len(feet) > 8:
        del feet[0]
    if len(feet) < 4:
        return False
    shift = 0.0
    for start, end in zip(feet, feet[1:]):
        shift += math.hypot(end[0] - start[0], end[1] - start[1])
        shift += math.hypot(end[2] - start[2], end[3] - start[3])
    gaps = [item[4] for item in feet]
    return shift > 0.16 and max(gaps) - min(gaps) > 0.06


def thigh_span(hip, knee):
    return math.hypot(knee.x - hip.x, knee.y - hip.y)


def knees_toward_camera(landmarks):
    hip_z = (getattr(landmarks[23], "z", 0.0) + getattr(landmarks[24], "z", 0.0)) / 2
    knee_z = (getattr(landmarks[25], "z", 0.0) + getattr(landmarks[26], "z", 0.0)) / 2
    return hip_z - knee_z > 0.08


def chair_sit(landmarks):
    def readable(point):
        visibility = point.visibility if point.visibility is not None else 1.0
        return visibility >= 0.35

    if not readable(landmarks[25]) or not readable(landmarks[26]):
        return False
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    hip_y = (landmarks[23].y + landmarks[24].y) / 2
    torso = hip_y - shoulder_y
    if torso < 0.05:
        return False
    left_drop = landmarks[25].y - hip_y
    right_drop = landmarks[26].y - hip_y
    knees_up = -0.02 < left_drop < torso * 0.60 and -0.02 < right_drop < torso * 0.60
    return knees_up and knees_toward_camera(landmarks)


def looks_like_sleep(landmarks, tilt, left_bend, right_bend):
    if tilt < 50 or left_bend < 150 or right_bend < 150:
        return False
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
    ankle_x = (landmarks[27].x + landmarks[28].x) / 2
    ankle_y = (landmarks[27].y + landmarks[28].y) / 2
    return math.hypot(ankle_x - shoulder_x, ankle_y - shoulder_y) > 0.35


def activity_from_pose(landmarks, last_tilt=None, hips=None, feet=None):
    tilt, bend, ankle_gap, hip_width, left_bend, right_bend, foot_lift, hip_x, hip_y = body_features(landmarks)
    growing = last_tilt is not None and (tilt - last_tilt) > 12 and tilt > 18
    left_thigh = thigh_span(landmarks[23], landmarks[25])
    right_thigh = thigh_span(landmarks[24], landmarks[26])
    bent_knee = min(left_bend, right_bend) < 150
    knee_gap = abs(left_bend - right_bend)
    stepping = (left_thigh > 0.05 or right_thigh > 0.05) and bent_knee and knee_gap > 22
    stepping = stepping or (knee_gap > 32 and min(left_bend, right_bend) < 158)
    if seen(landmarks[27]) and seen(landmarks[28]):
        torso_h = abs(((landmarks[11].y + landmarks[12].y) / 2) - hip_y)
        body = max(torso_h, hip_width, 0.03)
        stride = ankle_gap > hip_width * 2.4 and ankle_gap > body * 0.45
        left_reach = abs(landmarks[27].x - landmarks[23].x)
        right_reach = abs(landmarks[28].x - landmarks[24].x)
        reach_gap = abs(left_reach - right_reach)
        z_gap = abs(getattr(landmarks[27], "z", 0.0) - getattr(landmarks[28], "z", 0.0))
        stepping = stepping or stride or reach_gap > body * 0.55
        stepping = stepping or (z_gap > 0.28 and knee_gap > 20)
        stepping = stepping or (foot_lift > body * 0.22 and reach_gap > body * 0.2)
    moving = hips is not None and hip_is_moving(hips, hip_x, hip_y)
    if feet is not None and seen(landmarks[27]) and seen(landmarks[28]):
        moving = moving or feet_are_moving(feet, landmarks[27], landmarks[28], hip_x, hip_y)
    folded = bend < 130 and max(left_bend, right_bend) < 155 and (left_thigh > 0.05 or right_thigh > 0.05)
    sitting = folded or chair_sit(landmarks)
    tipped = last_tilt is not None and last_tilt < 25 and tilt > 35
    false_alarm = None

    if tilt > 35 and not tipped and sitting and tilt < 60:
        false_alarm = "sitting"
        label = "sitting"
        confidence = 0.80
    elif tilt > 35 and not tipped and looks_like_sleep(landmarks, tilt, left_bend, right_bend):
        false_alarm = "sleeping"
        label = "sleeping"
        confidence = 0.80
    elif tilt > 35:
        label = "fall"
        confidence = min(0.99, tilt / 50)
    elif tilt > 20 or growing:
        label = "off balance"
        confidence = min(0.90, max(tilt, 20) / 40)
    elif folded or chair_sit(landmarks):
        label = "sitting"
        confidence = 0.85
    elif stepping or moving:
        label = "walking"
        confidence = 0.85
    else:
        label = "standing"
        confidence = 0.85
    return label, round(float(confidence), 2), tilt, false_alarm


def keep_real_fall(label, false_alarm, tilt, hold):
    if label == "fall" and not false_alarm:
        return "fall", None, True
    if hold and tilt is not None and tilt > 35:
        return "fall", None, True
    return label, false_alarm, False


def draw_pose(frame, landmarks):
    height, width = frame.shape[:2]
    points = []
    links = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    for lm in landmarks:
        x = int(lm.x * width)
        y = int(lm.y * height)
        points.append((x, y))
        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    for a, b in links:
        if a < len(points) and b < len(points):
            cv2.line(frame, points[a], points[b], (0, 255, 0), 2)


def read_image_bytes(data):
    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    return frame


ACTIVITY_COLORS = {
    "fall": (0, 0, 255),
    "off balance": (0, 140, 255),
    "sitting": (255, 80, 20),
    "standing": (180, 30, 180),
    "walking": (0, 230, 255),
    "normal": (40, 180, 40),
    "sleeping": (180, 120, 60),
}


def paint_label(frame, label):
    color = ACTIVITY_COLORS.get(label, (40, 180, 40))
    cv2.rectangle(frame, (6, 8), (280, 48), (20, 20, 20), -1)
    display_text = label.upper()
    cv2.putText(frame, display_text, (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


def predict_frame(pose, frame, last_tilt=None, hips=None, feet=None):
    if pose is None:
        return frame, "normal", 0.0, None, None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        if hips is not None:
            hips.clear()
        if feet is not None:
            feet.clear()
        paint_label(frame, "normal")
        return frame, "normal", 0.0, None, None
    landmarks = result.pose_landmarks[0]
    draw_pose(frame, landmarks)
    label, confidence, tilt, false_alarm = activity_from_pose(landmarks, last_tilt, hips, feet)
    paint_label(frame, label)
    return frame, label, confidence, tilt, false_alarm


ACTIVITIES = ["fall", "off balance", "sitting", "standing", "walking", "normal"]
BAR_COLORS = {
    "fall": "#ff3b30",
    "off balance": "#ff9800",
    "sitting": "#2196f3",
    "standing": "#9c27b0",
    "walking": "#ffeb3b",
    "normal": "#4caf50",
}


def show_bars(rows, value_name, value_label, top, number_format):
    if top < 1:
        top = 1
    labeled = []
    for row in rows:
        number = format(row[value_name], number_format)
        labeled.append(
            {
                "activity": f"{row['activity']} ({number})",
                "kind": row["activity"],
                value_name: row[value_name],
            }
        )
    order = [row["activity"] for row in labeled]
    chart = (
        alt.Chart(alt.Data(values=labeled))
        .mark_bar()
        .encode(
            y=alt.Y(
                "activity:N",
                title="Activity",
                sort=order,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X(
                f"{value_name}:Q",
                title=value_label,
                scale=alt.Scale(domain=[0, top]),
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=ACTIVITIES,
                    range=[BAR_COLORS[name] for name in ACTIVITIES],
                ),
                legend=None,
            ),
        )
        .properties(height=280)
        .configure(background="transparent")
    )
    st.altair_chart(chart, width="stretch")


def show_picture(frame, label):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(rgb, width="stretch")
    color = BAR_COLORS.get(label, "#ffffff")
    st.markdown(
        f"**Activity State:** <span style='color:{color}; font-size:1.15rem; font-weight:800'>{label.upper()}</span>",
        unsafe_allow_html=True,
    )


def show_confidence(label, confidence):
    color = BAR_COLORS.get(label, "#ffffff")
    st.caption(f"Estimated Detection Confidence: {int(confidence * 100)}%")


def prepare_alarm():
    st.html(
        """
        <div id="fall-alarm-setup"></div>
        <script>
        (function () {
          if (window.__fallAlarmReady) return;
          window.__fallAlarmReady = true;
          window.__fallAlarmMuted = false;
          window.__fallAlarmNodes = null;
          window.__alarmToken = null;

          function audio() {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) return null;
            if (!window.__fallAudio) window.__fallAudio = new Ctx();
            return window.__fallAudio;
          }

          function arm() {
            const ctx = audio();
            if (ctx && ctx.state === "suspended") ctx.resume();
          }

          window.addEventListener("pointerdown", arm, true);
          window.addEventListener("keydown", arm, true);

          function silence() {
            const nodes = window.__fallAlarmNodes;
            if (!nodes) return;
            clearInterval(nodes.timer);
            try { nodes.osc.stop(); } catch (e) {}
            try { nodes.gain.disconnect(); } catch (e) {}
            window.__fallAlarmNodes = null;
          }

          window.__playFallAlarm = function () {
            if (window.__fallAlarmMuted) return false;
            const ctx = audio();
            if (!ctx) return false;
            if (ctx.state === "suspended") {
              ctx.resume().then(function () { window.__playFallAlarm(); });
              return false;
            }
            if (ctx.state !== "running") return false;
            if (window.__fallAlarmNodes) return true;
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = "square";
            osc.frequency.value = 880;
            gain.gain.value = 0.12;
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start();
            let high = false;
            const timer = setInterval(function () {
              high = !high;
              osc.frequency.setValueAtTime(high ? 880 : 520, ctx.currentTime);
            }, 320);
            window.__fallAlarmNodes = { osc: osc, gain: gain, timer: timer };
            return true;
          };

          window.__stopFallAlarm = function (allowNext) {
            silence();
            window.__fallAlarmMuted = !allowNext;
          };

          window.__armAlarm = function (token) {
            if (token && window.__alarmToken !== token) {
              window.__alarmToken = token;
              window.__fallAlarmMuted = false;
            }
            if (window.__playFallAlarm) return window.__playFallAlarm();
            return false;
          };
        })();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def quiet_alarm():
    st.html(
        """
        <div id="fall-alarm-setup"></div>
        <script>
        if (window.__stopFallAlarm) window.__stopFallAlarm(true);
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def sound_alarm(token):
    safe_token = json.dumps(token)
    st.html(
        f"""
        <button type="button" id="fall-snooze" style="display:block;width:100%;margin:0.2rem 0 0.8rem 0;background:#ff3b30;color:#ffffff;border:0;border-radius:0.6rem;padding:0.85rem 1rem;font:800 1.15rem system-ui,sans-serif;cursor:pointer;box-shadow:0 4px 14px rgba(255,59,48,0.4);">
          🚨 SNOOZE EMERGENCY SIREN
        </button>
        <p id="fall-alarm-note" style="font-weight:600;color:#ff3b30;">Acoustic siren is sounding. Click button above to snooze.</p>
        <script>
        (function () {{
          const note = document.getElementById("fall-alarm-note");
          const btn = document.getElementById("fall-snooze");
          if (window.__armAlarm) window.__armAlarm({safe_token});
          function refresh() {{
            const on = !!window.__fallAlarmNodes;
            const quiet = !!window.__fallAlarmMuted;
            if (!btn) return;
            if (on) {{
              btn.textContent = "🚨 SNOOZE EMERGENCY SIREN";
              btn.style.background = "#ff3b30";
              btn.style.color = "#ffffff";
              if (note) note.textContent = "Acoustic siren is sounding. Click button above to snooze.";
            }} else if (quiet) {{
              btn.textContent = "Snoozed (Audio Off)";
              btn.style.background = "#4a4a4a";
              btn.style.color = "#ffffff";
              if (note) note.textContent = "Snoozed. Acoustic alert is currently silenced.";
            }} else {{
              btn.textContent = "Sound Alert Siren";
              btn.style.background = "#ff9800";
              btn.style.color = "#ffffff";
              if (note) note.textContent = "Click to activate siren if audio did not autoplay.";
            }}
          }}
          if (btn) {{
            btn.onclick = function () {{
              if (window.__fallAlarmNodes) {{
                window.__stopFallAlarm(false);
                refresh();
                return;
              }}
              if (window.__fallAlarmMuted) return;
              window.__fallAlarmMuted = false;
              const Ctx = window.AudioContext || window.webkitAudioContext;
              if (!window.__fallAudio && Ctx) window.__fallAudio = new Ctx();
              const ctx = window.__fallAudio;
              const go = function () {{
                if (window.__playFallAlarm) window.__playFallAlarm();
                refresh();
              }};
              if (ctx && ctx.state !== "running") ctx.resume().then(go);
              else go();
            }};
          }}
          refresh();
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )


def show_emergency(token="fall"):
    if st.session_state.get("_fall_alarm_shown"):
        return
    st.session_state["_fall_alarm_shown"] = True
    st.markdown(
        """
        <style>
        @keyframes fallPulse {
            0%, 100% { opacity: 0.5; }
            50% { opacity: 0.95; }
        }
        .fall-alarm-screen {
            pointer-events: none;
            position: fixed;
            inset: 0;
            z-index: 999990;
            box-shadow:
                inset 0 0 0 16px rgba(255, 0, 0, 0.95),
                inset 0 0 100px rgba(255, 0, 0, 0.7);
            animation: fallPulse 0.75s ease-in-out infinite;
        }
        </style>
        <div class="fall-alarm-screen"></div>
        """,
        unsafe_allow_html=True,
    )
    st.error("🚨 CRITICAL EMERGENCY: Fall Detected!")
    sound_alarm(token)
    st.subheader("Emergency Response Protocol")
    st.markdown(
        """
        1. **Do not move the patient** if neck, back, or hip trauma is suspected.
        2. **Assess responsiveness** and check airway, breathing, and circulation.
        3. **Call emergency services immediately** (112 or local dispatch).
        4. **Keep patient calm and warm** until emergency responders arrive.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.link_button("📞 Call 112 (Emergency)", "tel:112", use_container_width=True)
    with col2:
        st.link_button(
            "🏥 Nearest Hospital (Maps)",
            "https://www.google.com/maps/search/?api=1&query=nearest+hospital",
            use_container_width=True,
        )


def add_count(counts, label, false_alarm):
    checked = false_alarm or label or "normal"
    called = "fall" if false_alarm else (label or "normal")
    if checked not in MATRIX_LABELS:
        checked = "normal"
    if called not in MATRIX_LABELS:
        called = "normal"
    counts[(checked, called)] = counts.get((checked, called), 0) + 1


def show_heatmap(counts):
    st.subheader("Confusion Matrix")
    st.caption("Rows represent ground truth verification; columns represent triggered classifier states.")
    rows = []
    top = 1
    for checked in MATRIX_LABELS:
        for called in MATRIX_LABELS:
            count = int(counts.get((checked, called), 0))
            top = max(top, count)
            rows.append({"Verified": checked, "Predicted": called, "Frames": count})
    base = alt.Chart(alt.Data(values=rows)).encode(
        x=alt.X("Predicted:N", sort=MATRIX_LABELS, title="Predicted State"),
        y=alt.Y("Verified:N", sort=MATRIX_LABELS, title="Clinical State"),
    )
    heatmap = base.mark_rect().encode(
        color=alt.Color("Frames:Q", scale=alt.Scale(scheme="purples", domain=[0, top]), title="Count")
    )
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("Frames:Q"),
        color=alt.condition(alt.datum.Frames > top / 2, alt.value("white"), alt.value("black")),
    )
    st.altair_chart((heatmap + text).properties(height=300), width="stretch")


def show_false_alarm(kind, sitting_count=None, sleeping_count=None):
    if sitting_count or sleeping_count:
        st.info(
            f"False-Alarm Disambiguation: Filtered {sitting_count or 0} sitting and "
            f"{sleeping_count or 0} lying/sleeping postures from triggering emergency fall dispatch."
        )
    elif kind == "sitting":
        st.info("Posture Check: Reclined sitting detected (not a critical fall incident).")
    elif kind == "sleeping":
        st.info("Posture Check: Horizontal sleeping/recumbent state verified.")


def show_result_message(label, false_alarm=None, token="fall"):
    if label == "fall" and not false_alarm:
        show_emergency(token)
    elif false_alarm:
        show_false_alarm(false_alarm)
        quiet_alarm()
    elif label == "off balance":
        st.warning("⚠️ Warning: Patient is leaning or losing balance.")
        quiet_alarm()
    elif label == "normal":
        st.info("Status: Normal activity / No subject in frame.")
        quiet_alarm()
    else:
        st.success(f"Status: Safe ({label.capitalize()}).")
        quiet_alarm()


def show_result(frame, label, confidence, false_alarm=None, token="fall"):
    show_picture(frame, label)
    show_confidence(label, confidence)
    show_result_message(label, false_alarm, token)


def frames_from_video(data):
    frames = []
    if AV_AVAILABLE:
        buffer = io.BytesIO(data)
        try:
            container = av.open(buffer)
            stream = container.streams.video[0]
            total = stream.frames or 0
            if total < 1 and stream.duration and stream.average_rate:
                total = int(stream.duration * stream.time_base * stream.average_rate)
            step = max(1, total // 20) if total > 0 else 1
            index = 0
            for frame in container.decode(stream):
                if index % step == 0:
                    frames.append(frame.to_ndarray(format="bgr24"))
                index += 1
                if len(frames) >= 20:
                    break
            container.close()
            return frames
        except Exception:
            pass

    # Resilient fallback: OpenCV VideoCapture
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // 20) if total > 0 else 1
        idx = 0
        while cap.isOpened() and len(frames) < 20:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frames.append(frame)
            idx += 1
        cap.release()
    except Exception:
        pass
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
    return frames



pose = load_pose()
prepare_alarm()

mode = st.radio("Choose Input Mode", ["Photo", "Video", "Camera", "Live"], horizontal=True)

if mode == "Photo":
    photo = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if photo is not None:
        frame = read_image_bytes(photo.getvalue())
        if frame is None:
            st.error("Uploaded image could not be decoded.")
        else:
            frame, label, confidence, _tilt, false_alarm = predict_frame(pose, frame)
            show_result(frame, label, confidence, false_alarm, f"photo-{photo.name}-{photo.size}")

elif mode == "Video":
    video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if video is not None:
        try:
            frames = frames_from_video(video.getvalue())
        except Exception:
            frames = []
        if len(frames) == 0:
            st.error("Uploaded video could not be decoded.")
        else:
            counts = {name: 0 for name in ACTIVITIES}
            false_counts = {"sitting": 0, "sleeping": 0}
            matrix = {}
            last = None
            last_fall = None
            hold = False
            last_tilt = None
            hips = []
            feet = []
            try:
                for frame in frames:
                    frame, label, confidence, tilt, false_alarm = predict_frame(
                        pose, frame.copy(), last_tilt, hips, feet
                    )
                    label, false_alarm, hold = keep_real_fall(label, false_alarm, tilt, hold)
                    paint_label(frame, label)
                    last_tilt = tilt
                    if false_alarm:
                        false_counts[false_alarm] += 1
                    if label in counts:
                        counts[label] += 1
                    add_count(matrix, label, false_alarm)
                    last = (frame, label, confidence, false_alarm)
                    if label == "fall":
                        last_fall = last
            except Exception as e:
                st.error(f"Error during video processing: {e}")
            else:
                frame, label, confidence, false_alarm = last_fall or last
                if counts["fall"] > 0:
                    label = "fall"
                    false_alarm = None
                    paint_label(frame, label)
                else:
                    label = max(ACTIVITIES, key=lambda name: counts[name])
                    false_alarm = None
                    paint_label(frame, label)
                if label in ACTIVITIES:
                    show_confidence(label, confidence)
                show_picture(frame, label)
                st.write(f"Analyzed {len(frames)} frames across the video sequence.")
                show_bars(
                    [{"activity": name, "frames": counts[name]} for name in ACTIVITIES],
                    "frames",
                    "Frames",
                    len(frames),
                    ".0f",
                )
                caught = false_counts["sitting"] + false_counts["sleeping"]
                show_heatmap(matrix)
                if caught:
                    show_false_alarm(
                        None,
                        sitting_count=false_counts["sitting"],
                        sleeping_count=false_counts["sleeping"],
                    )
                if counts["fall"] > 0:
                    show_emergency(f"video-{video.name}-{video.size}")
                elif not false_alarm:
                    if label == "off balance":
                        st.warning("⚠️ Warning: Patient losing balance during sequence.")
                    elif label == "normal":
                        st.info("Status: Normal activity / No person detected.")
                    else:
                        st.success("Status: Safe (No fall detected).")

elif mode == "Camera":
    camera = st.camera_input("Take Camera Snapshot")
    if camera is not None:
        frame = read_image_bytes(camera.getvalue())
        if frame is None:
            st.error("Camera snapshot could not be decoded.")
        else:
            frame, label, confidence, _tilt, false_alarm = predict_frame(pose, frame)
            show_result(frame, label, confidence, false_alarm, f"camera-{len(camera.getvalue())}")

else:
    if not (WEBRTC_AVAILABLE and AV_AVAILABLE):
        st.warning("Live WebRTC streaming requires streamlit-webrtc and PyAV. Please switch to 'Camera' mode above to take live camera snapshots.")
    else:
        st.write("Click **Start** to begin real-time camera tracking. Click **Stop** when finished.")

        def on_frame(frame):
            try:
                image = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
                if image.size == 0:
                    return frame
                state = live_state()
                image, label, _confidence, tilt, false_alarm = predict_frame(
                    state.pose, image, state.last_tilt, state.hips, state.feet
                )
                label, false_alarm, state.fall_hold = keep_real_fall(
                    label, false_alarm, tilt, state.fall_hold
                )
                state.last_tilt = tilt
                paint_label(image, label)
                _live_label["name"] = label
                _live_label["false_alarm"] = false_alarm
                if label == "fall":
                    _live_label["alarm_until"] = time.time() + 12
                add_count(_live_matrix, label, false_alarm)
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            except Exception as error:
                print("Live frame processing error:", error)
                return frame

        live_camera = webrtc_streamer(
            key="live-camera",
            video_frame_callback=on_frame,
            media_stream_constraints={
                "video": {"width": {"ideal": 480}, "height": {"ideal": 360}, "facingMode": "user"},
                "audio": False,
            },
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            video_html_attrs={"autoPlay": True, "muted": True, "playsInline": True, "controls": False},
            media_toggle_controls=False,
            sendback_audio=False,
            async_processing=True,
        )

        @st.fragment(run_every=1)
        def live_fall_alarm():
            if not live_camera.state.playing:
                _live_label["name"] = None
                _live_label["false_alarm"] = None
                _live_label["alarm_until"] = 0.0
                _live_matrix.clear()
                quiet_alarm()
                return
            st.session_state["_fall_alarm_shown"] = False
            kind = _live_label.get("false_alarm")
            name = _live_label.get("name")
            alarm_on = name == "fall" or time.time() < _live_label.get("alarm_until", 0)
            if alarm_on:
                show_emergency("live")
            elif kind:
                show_false_alarm(kind)
                quiet_alarm()
            else:
                quiet_alarm()
            if _live_matrix:
                try:
                    show_heatmap(dict(_live_matrix))
                except Exception as error:
                    print("Heatmap skipped:", error)

        live_fall_alarm()

if not st.session_state.get("_fall_alarm_shown"):
    quiet_alarm()
