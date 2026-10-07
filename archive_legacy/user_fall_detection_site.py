# Fall detection site.
# The activity comes from the body: shoulders, hips, knees, and feet.

import io
import math
import threading
from pathlib import Path

import altair as alt
import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer

ROOT = Path(__file__).parent
TASK_PATH = ROOT / "pose_landmarker.task"
if not TASK_PATH.exists():
    TASK_PATH = ROOT / "model" / "pose_landmarker_lite.task"
LOGO_PATH = ROOT / "logo.png"

st.set_page_config(page_title="Fall detection", layout="centered")


def current_theme():
    # The address bar remembers the choice after the page reloads.
    saved = st.query_params.get("theme")
    if saved in ("light", "dark"):
        return saved
    detected = st.context.theme.type
    if detected in ("light", "dark"):
        return detected
    return "dark"


def paint_page(theme_name):
    # Same colour bloom in both modes. Light uses a pale base, dark uses the night base.
    if theme_name == "light":
        base = "#fff8f6"
        blooms = """
            radial-gradient(circle at 8% 0%, rgba(255, 59, 48, 0.55), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 152, 0, 0.50), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(33, 150, 243, 0.48), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(156, 39, 176, 0.50), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 59, 0.42), transparent 40%),
            radial-gradient(circle at 72% 58%, rgba(76, 175, 80, 0.42), transparent 44%)
        """
    else:
        base = "#3a2460"
        blooms = """
            radial-gradient(circle at 18% 12%, rgba(150, 230, 230, 0.55), transparent 26%),
            radial-gradient(circle at 8% 0%, rgba(255, 110, 100, 0.72), transparent 46%),
            radial-gradient(circle at 96% 6%, rgba(255, 176, 70, 0.62), transparent 44%),
            radial-gradient(circle at 100% 78%, rgba(90, 180, 255, 0.58), transparent 48%),
            radial-gradient(circle at 0% 86%, rgba(190, 110, 230, 0.62), transparent 46%),
            radial-gradient(circle at 48% 42%, rgba(255, 235, 120, 0.28), transparent 36%),
            radial-gradient(circle at 72% 58%, rgba(120, 220, 130, 0.42), transparent 42%)
        """
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {base};
            background-image: {blooms};
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{
            background: transparent;
        }}
        [data-testid="stMainMenuDivider"]:has(+ [data-testid="stThemeSwitcher"]),
        [data-testid="stThemeSwitcher"] {{
            display: none;
        }}
        /* The logo's built-in expand button only grows the picture. */
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stElementToolbar"] {{
            display: none;
        }}
        /* The about icon sits on the logo and only shows while the pointer is there. */
        [data-testid="stColumn"]:has([data-testid="stImage"]) {{
            position: relative;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stVerticalBlock"] {{
            gap: 0;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
            position: absolute;
            top: 38px;
            left: 40px;
            width: 2.15rem;
            height: 2.15rem;
            margin: 0;
            z-index: 5;
            opacity: 0;
            transition: opacity 0.15s ease;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]):hover [data-testid="stElementContainer"]:has([data-testid="stButton"]) {{
            opacity: 1;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stBaseButton-tertiary"] {{
            width: 2.15rem;
            min-height: 2.15rem;
            padding: 0;
            border-radius: 999px;
            background: #ffffff;
            color: #1a2744;
            border: 1px solid rgba(26, 39, 68, 0.18);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.35);
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stIconMaterial"] {{
            font-size: 1.2rem;
        }}
        [data-testid="stColumn"]:has([data-testid="stImage"]) [data-testid="stBaseButton-tertiary"] p {{
            display: none;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def switch_theme(choice):
    # Streamlit reads this saved name on the next page load.
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


@st.dialog("What this AI does")
def show_about():
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=120)
    st.write(
        "This site looks at a person's body and names what they are doing. "
        "It uses the shoulders, hips, knees, and feet."
    )
    st.write("You can give it a photo, a video, one camera shot, or the live camera.")
    st.markdown(
        """
- **Fall** — the body has tipped over
- **Off balance** — the body is leaning
- **Sitting**
- **Standing**
- **Walking**
- **Normal** — no person is in the picture
        """
    )
    st.write("A photo or a camera shot gives one answer. A video counts the frames and draws the charts.")


main_col, theme_col = st.columns([4.4, 1.3], vertical_alignment="center")
with main_col:
    logo_col, title_col = st.columns([1, 4.5], vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=78)
        if st.button("About", icon=":material/info:", key="logo_about", type="tertiary"):
            show_about()
    with title_col:
        st.title("Fall detection")
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

st.write("Upload a photo or a video, take one picture, or start the live camera.")


def make_pose():
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


# The live callback runs on its own thread. MediaPipe has to stay on that thread.
_live_local = threading.local()


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
    # A missing foot should not count as a step.
    visibility = point.visibility if point.visibility is not None else 1.0
    return visibility >= 0.5


def body_features(landmarks):
    def get_c(pt, attr):
        if hasattr(pt, attr):
            return getattr(pt, attr)
        elif isinstance(pt, (list, tuple, np.ndarray)):
            idx_map = {"x": 0, "y": 1, "z": 2, "visibility": 3}
            i = idx_map.get(attr, 0)
            return float(pt[i]) if len(pt) > i else 0.0
        return 0.0

    sh_lx, sh_ly = get_c(landmarks[11], "x"), get_c(landmarks[11], "y")
    sh_rx, sh_ry = get_c(landmarks[12], "x"), get_c(landmarks[12], "y")
    shoulder_x = (sh_lx + sh_rx) / 2.0
    shoulder_y = (sh_ly + sh_ry) / 2.0
    
    hip_lx, hip_ly = get_c(landmarks[23], "x"), get_c(landmarks[23], "y")
    hip_rx, hip_ry = get_c(landmarks[24], "x"), get_c(landmarks[24], "y")
    hip_x = (hip_lx + hip_rx) / 2.0
    hip_y = (hip_ly + hip_ry) / 2.0
    
    left_bend = knee_bend(landmarks[23], landmarks[25], landmarks[27])
    right_bend = knee_bend(landmarks[24], landmarks[26], landmarks[28])
    bend = (left_bend + right_bend) / 2.0
    min_bend = min(left_bend, right_bend)
    knee_asym = abs(left_bend - right_bend)
    
    ank_lx, ank_ly = get_c(landmarks[27], "x"), get_c(landmarks[27], "y")
    ank_rx, ank_ry = get_c(landmarks[28], "x"), get_c(landmarks[28], "y")
    ankle_gap = abs(ank_lx - ank_rx)
    hip_width = abs(hip_lx - hip_rx)
    foot_lift = abs(ank_ly - ank_ry)
    
    dx = hip_x - shoulder_x
    dy = hip_y - shoulder_y
    tilt_2d = abs(math.degrees(math.atan2(dx, dy))) if (dx or dy) else 0.0
    
    sh_lz, sh_rz = get_c(landmarks[11], "z"), get_c(landmarks[12], "z")
    hip_lz, hip_rz = get_c(landmarks[23], "z"), get_c(landmarks[24], "z")
    dz = ((hip_lz + hip_rz) / 2.0) - ((sh_lz + sh_rz) / 2.0)
    norm_3d = math.hypot(dx, dy, dz) + 1e-7
    cos_3d = max(-1.0, min(1.0, dy / norm_3d))
    tilt_3d = math.degrees(math.acos(cos_3d))
    tilt = max(tilt_2d, tilt_3d)
    
    xs = [get_c(lm, "x") for lm in landmarks]
    ys = [get_c(lm, "y") for lm in landmarks]
    w = max(xs) - min(xs) + 1e-5
    h = max(ys) - min(ys) + 1e-5
    aspect_ratio = w / h
    cog_y = hip_y
    
    bos_min_x = min(ank_lx, ank_rx)
    bos_max_x = max(ank_lx, ank_rx)
    bos_width = max(0.06, bos_max_x - bos_min_x)
    bos_center_x = (bos_min_x + bos_max_x) / 2.0
    com_x = 0.55 * hip_x + 0.45 * shoulder_x
    balance_deviation = abs(com_x - bos_center_x)
    balance_ratio = balance_deviation / (0.5 * bos_width + 1e-5)
    
    ground_y = max(ys)
    hip_clearance = ground_y - hip_y

    return {
        "tilt": tilt,
        "bend": bend,
        "min_bend": min_bend,
        "knee_asym": knee_asym,
        "ankle_gap": ankle_gap,
        "hip_width": hip_width,
        "foot_lift": foot_lift,
        "hip_x": hip_x,
        "hip_y": hip_y,
        "aspect_ratio": aspect_ratio,
        "cog_y": cog_y,
        "balance_ratio": balance_ratio,
        "hip_clearance": hip_clearance
    }


def hip_is_moving(hips, hip_x, hip_y):
    # A walk toward the camera keeps the feet close, but the hips still travel.
    hips.append((hip_x, hip_y))
    if len(hips) > 10:
        del hips[0]
    if len(hips) < 6:
        return False
    path = 0.0
    for start, end in zip(hips, hips[1:]):
        path += math.hypot(end[0] - start[0], end[1] - start[1])
    net = math.hypot(hips[-1][0] - hips[0][0], hips[-1][1] - hips[0][1])
    # Shifting weight stays in place. A walk keeps moving the same way.
    return path > 0.10 and net > path * 0.45


def thigh_span(hip, knee):
    return math.hypot(knee.x - hip.x, knee.y - hip.y)


def knees_toward_camera(landmarks):
    # MediaPipe z is smaller when a point is closer to the camera.
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
    return knees_up or knees_toward_camera(landmarks)


def activity_from_pose(landmarks, last_tilt=None, hips=None):
    f = body_features(landmarks)
    tilt = f["tilt"]
    bend = f["bend"]
    min_bend = f["min_bend"]
    knee_asym = f["knee_asym"]
    ankle_gap = f["ankle_gap"]
    hip_width = f["hip_width"]
    foot_lift = f["foot_lift"]
    aspect_ratio = f["aspect_ratio"]
    cog_y = f["cog_y"]
    balance_ratio = f["balance_ratio"]
    hip_clearance = f["hip_clearance"]
    hip_x, hip_y = f["hip_x"], f["hip_y"]

    growing = last_tilt is not None and (tilt - last_tilt) > 12 and tilt > 18
    
    # 1. Floor Collapse Fail-Safe (Clear fall on ground)
    is_floor_collapse = (
        (tilt >= 60.0 and aspect_ratio >= 0.75 and cog_y > 0.58) or
        (tilt >= 70.0 and (aspect_ratio >= 0.65 or cog_y > 0.60)) or
        (aspect_ratio >= 0.95 and tilt > 45.0 and cog_y > 0.55)
    )
    
    # 2. Sitting Indicators (Must have flexed knees and elevated hips)
    is_seated_angles = (bend < 138.0 or min_bend < 132.0) and bend < 150.0
    is_sit = is_seated_angles and (aspect_ratio < 0.78 and cog_y < 0.85) and tilt < 58.0 and not is_floor_collapse
    
    # 3. Walking Indicators (Dynamic stride, knee asymmetry, foot lift)
    has_stride = (ankle_gap > max(0.06, hip_width * 2.2))
    has_knee_stride = (knee_asym > 15.0)
    has_foot_lift = (foot_lift > 0.035)
    moving = hips is not None and hip_is_moving(hips, hip_x, hip_y)
    is_walking = (has_stride or has_knee_stride or has_foot_lift or moving) and tilt < 22.0 and not is_sit and not is_floor_collapse
    
    # 4. Controlled Bending (Normal Activity - requires supporting extended legs and counterbalanced feet)
    is_controlled_bend = (
        (24.0 <= tilt <= 75.0) and
        (aspect_ratio < 0.90) and
        (hip_clearance > 0.16) and
        (balance_ratio <= 1.25) and
        (bend >= 140.0 and min_bend >= 130.0) and
        not is_floor_collapse
    )

    # 5. Off-Balance Instability (Stumble / Pre-Fall)
    is_off_balance = (
        (tilt >= 20.0 or (tilt >= 15.0 and balance_ratio > 1.25) or growing) and
        not is_sit and
        not is_floor_collapse and
        not is_controlled_bend and
        not is_walking and
        aspect_ratio < 1.10
    )

    # Classification Hierarchy
    if is_floor_collapse or tilt > 65.0:
        label = "fall"
        confidence = min(0.99, max(0.90, tilt / 70.0))
    elif is_sit:
        label = "sitting"
        confidence = 0.92
    elif is_controlled_bend:
        label = "normal"
        confidence = 0.88
    elif is_walking:
        label = "walking"
        confidence = 0.88
    elif is_off_balance:
        label = "off balance"
        confidence = min(0.95, max(0.85, max(tilt, 20) / 40.0))
    elif tilt < 20.0 and bend > 155.0:
        label = "standing"
        confidence = 0.90
    else:
        label = "standing"
        confidence = 0.85

    return label, round(float(confidence), 2), tilt


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


# OpenCV uses blue, green, red order.
ACTIVITY_COLORS = {
    "fall": (0, 0, 255),
    "off balance": (0, 140, 255),
    "sitting": (255, 80, 20),
    "standing": (180, 30, 180),
    "walking": (0, 230, 255),
    "normal": (40, 180, 40),
}


def paint_label(frame, label):
    color = ACTIVITY_COLORS.get(label, (40, 180, 40))
    cv2.putText(frame, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def predict_frame(pose, frame, last_tilt=None, hips=None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = pose.detect(mp_image)
    if not result.pose_landmarks:
        if hips is not None:
            hips.clear()
        paint_label(frame, "normal")
        return frame, "normal", 0.0, None
    landmarks = result.pose_landmarks[0]
    draw_pose(frame, landmarks)
    label, confidence, tilt = activity_from_pose(landmarks, last_tilt, hips)
    paint_label(frame, label)
    return frame, label, confidence, tilt


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
    # The axis stops at the real total, and every activity fits on the page.
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
        f"Activity: <span style='color:{color}; font-weight:700'>{label}</span>",
        unsafe_allow_html=True,
    )


def show_confidence(label, confidence):
    show_bars(
        [
            {"activity": name, "confidence": confidence if name == label else 0}
            for name in ACTIVITIES
        ],
        "confidence",
        "Confidence",
        1,
        ".2f",
    )


def show_result_message(label):
    if label == "fall":
        st.error("EMERGENCY: Fall detected. Please check on the person.")
    elif label == "off balance":
        st.warning("Warning: the person is losing balance.")
    elif label == "normal":
        st.info("No person found in this frame.")
    else:
        st.success("No fall detected.")


def show_result(frame, label, confidence):
    show_picture(frame, label)
    show_confidence(label, confidence)
    show_result_message(label)


def frames_from_video(data):
    # Read a few frames from the upload. A long video stays quick.
    # PyAV reads the file in memory, so the app folder does not need to be writable.
    frames = []
    try:
        container = av.open(io.BytesIO(data))
    except Exception:
        return frames
    try:
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
    except Exception:
        return frames
    finally:
        container.close()
    return frames


pose = load_pose()
mode = st.radio("Choose an input", ["Photo", "Video", "Camera", "Live"], horizontal=True)

if mode == "Photo":
    photo = st.file_uploader("Photo", type=["jpg", "jpeg", "png"])
    if photo is not None:
        frame = read_image_bytes(photo.getvalue())
        if frame is None:
            st.error("That file could not be read as a photo.")
        else:
            frame, label, confidence, _tilt = predict_frame(pose, frame)
            show_result(frame, label, confidence)

elif mode == "Video":
    video = st.file_uploader("Video", type=["mp4", "avi", "mov"])
    if video is not None:
        try:
            frames = frames_from_video(video.getvalue())
        except Exception:
            frames = []
        if len(frames) == 0:
            st.error("That video could not be read.")
        else:
            counts = {name: 0 for name in ACTIVITIES}
            last = None
            try:
                for frame in frames:
                    frame, label, confidence, _tilt = predict_frame(pose, frame.copy())
                    counts[label] += 1
                    last = (frame, label, confidence)
            except Exception:
                st.error("That video could not be read.")
            else:
                frame, label, confidence = last
                show_confidence(label, confidence)
                show_picture(frame, label)
                st.write("Frames checked:", len(frames))
                show_bars(
                    [{"activity": name, "frames": counts[name]} for name in ACTIVITIES],
                    "frames",
                    "Frames",
                    len(frames),
                    ".0f",
                )
                show_result_message(label)
                if counts["fall"] > 0:
                    st.error("EMERGENCY: A fall showed up in this video.")

elif mode == "Camera":
    camera = st.camera_input("Take a picture")
    if camera is not None:
        frame = read_image_bytes(camera.getvalue())
        if frame is None:
            st.error("The camera picture could not be read.")
        else:
            frame, label, confidence, _tilt = predict_frame(pose, frame)
            show_result(frame, label, confidence)

else:
    st.write("Press Start and allow the camera. Click Stop when you are done.")
    st.write("If the picture stays black, press Stop, refresh this page, then press Start again.")

    def on_frame(frame):
        # Always send a picture back. A pose error must not turn the camera black.
        try:
            image = np.ascontiguousarray(frame.to_ndarray(format="bgr24"))
            if image.size == 0:
                return frame
            state = live_state()
            image, _label, _confidence, state.last_tilt = predict_frame(
                state.pose, image, state.last_tilt, state.hips
            )
            return av.VideoFrame.from_ndarray(image, format="bgr24")
        except Exception as error:
            print("Live frame skipped:", error)
            return frame

    webrtc_streamer(
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
