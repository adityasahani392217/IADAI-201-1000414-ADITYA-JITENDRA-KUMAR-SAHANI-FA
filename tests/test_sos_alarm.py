"""
Verification test for escalating alarm synthesizer, Google Maps redirects,
and SOS auto-calling components.
"""
import sys
from pathlib import Path

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import ui.components as c
import ui.dashboard_view as d
import ui.styles as s
from utils.alert_manager import AlertManager

def test_siren_synthesis():
    for stage in [1, 2, 3]:
        wav_b64 = c.synthesize_acoustic_siren_base64(stage)
        assert len(wav_b64) > 100, f"Stage {stage} WAV should be non-empty"
    print("PASS: Acoustic siren synthesis stages 1, 2, 3")

def test_escalating_alarm_html():
    h1 = c.render_escalating_alarm_synthesizer(4.0)
    assert "Stage 1" in h1, "Should identify Stage 1 (<10s)"
    h2 = c.render_escalating_alarm_synthesizer(16.0)
    assert "Stage 2" in h2, "Should identify Stage 2 (10-25s)"
    h3 = c.render_escalating_alarm_synthesizer(30.0)
    assert "Stage 3" in h3, "Should identify Stage 3 (>25s)"
    print("PASS: Escalating alarm synthesizer HTML")

def test_hospital_and_sos():
    hosp_html = c.render_hospital_locator_cards()
    assert "google.com/maps" in hosp_html
    assert "St. Jude Metropolitan" in hosp_html
    print("PASS: Hospital locator cards with Google Maps links")

    sos_html = c.render_sos_countdown_html(15, 12.0)
    assert "tel:911" in sos_html
    print("PASS: SOS auto-call countdown with tel:911")

    speed_dial = c.render_speed_dial_list_html()
    assert "Dr. Sarah Mitchell" in speed_dial
    print("PASS: Speed dial directory with emergency contacts")

    sim_call_html = c.render_simulated_calling_screen_html("Senior Resident A", "FALL-SIM-911", "Room 01")
    assert "CLINICAL TRAINING SIMULATION MODE" in sim_call_html
    assert "911 EMS Medical Dispatch" in sim_call_html
    assert "Dispatcher Voice Transcript" in sim_call_html
    assert "speechSynthesis" in sim_call_html
    print("PASS: Simulated 911 calling screen with recorded voice and simulation disclosure")

def test_alert_manager():
    am = AlertManager()
    res = am.trigger_fall_alert(
        confidence=0.97,
        metrics={"torso_angle_deg": 85.0, "aspect_ratio": 0.40},
        patient_id="TEST-PATIENT",
        room_loc="Living Room"
    )
    assert res["status"] == "DISPATCH_SENT"
    print(f"PASS: AlertManager triggered fall alert incident: {res['incident_id']}")

if __name__ == "__main__":
    test_siren_synthesis()
    test_escalating_alarm_html()
    test_hospital_and_sos()
    test_alert_manager()
    print("ALL TESTS PASSED SUCCESSFULLY!")

