from types import SimpleNamespace

from cereal import log
from openpilot.common.realtime import DT_DMON
from openpilot.selfdrive.monitoring.policy import DriverMonitoring, DRIVER_MONITOR_SETTINGS, starpilot_aol_enabled

EventName = log.OnroadEvent.EventName
dm_settings = DRIVER_MONITOR_SETTINGS()

TEST_TIMESPAN = 120  # seconds
DISTRACTED_SECONDS_TO_ORANGE = dm_settings._VISION_POLICY_ALERT_2_TIMEOUT + 1
DISTRACTED_SECONDS_TO_RED = dm_settings._VISION_POLICY_ALERT_3_TIMEOUT + 1
INVISIBLE_SECONDS_TO_ORANGE = dm_settings._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT + 1
INVISIBLE_SECONDS_TO_RED = dm_settings._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT + 1

def make_msg(face_detected, distracted=False, model_uncertain=False):
  ds = log.DriverStateV2.new_message()
  ds.leftDriverData.faceOrientation = [0., 0., 0.]
  ds.leftDriverData.facePosition = [0., 0.]
  ds.leftDriverData.faceProb = 1. * face_detected
  ds.leftDriverData.leftEyeProb = 1.
  ds.leftDriverData.rightEyeProb = 1.
  ds.leftDriverData.leftBlinkProb = 1. * distracted
  ds.leftDriverData.rightBlinkProb = 1. * distracted
  ds.leftDriverData.faceOrientationStd = [1.*model_uncertain, 1.*model_uncertain, 1.*model_uncertain]
  ds.leftDriverData.facePositionStd = [1.*model_uncertain, 1.*model_uncertain]
  # TODO: test both separately when e2e is used
  ds.leftDriverData.phoneProb = 0.
  return ds


def make_msg_eyelids(closure):
  # heavy eyelids / microsleep: partial eye closure, possibly below the
  # stock full-blink threshold
  ds = make_msg(True)
  ds.leftDriverData.leftBlinkProb = closure
  ds.leftDriverData.rightBlinkProb = closure
  return ds


def make_msg_pitch(pitch):
  # head pitch in radians (negative = tilted down/forward)
  ds = make_msg(True)
  ds.leftDriverData.faceOrientation = [pitch, 0., 0.]
  return ds


# driver state from neural net, 10Hz
msg_NO_FACE_DETECTED = make_msg(False)
msg_ATTENTIVE = make_msg(True)
msg_DISTRACTED = make_msg(True, distracted=True)
msg_ATTENTIVE_UNCERTAIN = make_msg(True, model_uncertain=True)
msg_DISTRACTED_UNCERTAIN = make_msg(True, distracted=True, model_uncertain=True)
msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN = make_msg(True, distracted=True, model_uncertain=dm_settings._HI_STD_THRESHOLD*1.5)

# driver interaction with car
car_interaction_DETECTED = True
car_interaction_NOT_DETECTED = False

# some common state vectors
always_no_face = [msg_NO_FACE_DETECTED] * int(TEST_TIMESPAN / DT_DMON)
always_attentive = [msg_ATTENTIVE] * int(TEST_TIMESPAN / DT_DMON)
always_distracted = [msg_DISTRACTED] * int(TEST_TIMESPAN / DT_DMON)
always_true = [True] * int(TEST_TIMESPAN / DT_DMON)
always_false = [False] * int(TEST_TIMESPAN / DT_DMON)


class FakeSubMaster:
  def __init__(self, aol_enabled, alive=True, valid=True):
    self.data = {'starpilotCarState': SimpleNamespace(alwaysOnLateralEnabled=aol_enabled)}
    self.alive = {'starpilotCarState': alive}
    self.valid = {'starpilotCarState': valid}

  def __getitem__(self, service):
    return self.data[service]


class TestMonitoring:
  def _run_seq(self, msgs, interaction, engaged, lowspeed):
    DM = DriverMonitoring()
    alert_lvls = []
    for idx in range(len(msgs)):
      DM._update_states(msgs[idx], [0, 0, 0], 0, engaged[idx], lowspeed[idx])
      # cal_rpy and car_speed don't matter here

      # evaluate events at 10Hz for tests
      DM._update_events(interaction[idx], engaged[idx], lowspeed[idx], 0)
      alert_lvls.append(DM.alert_level)
    assert len(alert_lvls) == len(msgs), f"got {len(alert_lvls)} for {len(msgs)} driverState input msgs"
    return alert_lvls, DM


  def test_rhd_manual_override_beats_saved_default(self):
    DM = DriverMonitoring(rhd_saved=False, rhd_override=True)
    DM._update_states(msg_ATTENTIVE, [0, 0, 0], 0, False, False)
    assert DM.wheel_on_right

    DM = DriverMonitoring(rhd_saved=True, rhd_override=False)
    DM._update_states(msg_ATTENTIVE, [0, 0, 0], 0, False, False)
    assert not DM.wheel_on_right


  def test_starpilot_aol_counts_as_dm_engaged(self):
    assert starpilot_aol_enabled(FakeSubMaster(True))
    assert not starpilot_aol_enabled(FakeSubMaster(False))
    assert not starpilot_aol_enabled(FakeSubMaster(True, alive=False))
    assert not starpilot_aol_enabled(FakeSubMaster(True, valid=False))


  # engaged, driver is attentive all the time
  def test_fully_aware_driver(self):
    alert_lvls, d_status = self._run_seq(always_attentive, always_false, always_true, always_false)
    assert all(a == 0 for a in alert_lvls)
    assert d_status.active_policy == log.DriverMonitoringState.MonitoringPolicy.vision

  # engaged, driver is distracted and does nothing
  def test_fully_distracted_driver(self):
    alert_lvls, d_status = self._run_seq(always_distracted, always_false, always_true, always_false)
    # drowsiness fast-path: sustained eye closure trips drowsy at ~3s, then
    # awareness drains 2x -> alert_1 ~4.5s, alert_2 ~6s, red ~8.5s (stock: 5s/8s/13s)
    assert alert_lvls[int(2 / DT_DMON)] == 0
    assert alert_lvls[int(4.75 / DT_DMON)] == 1
    assert alert_lvls[int(6.75 / DT_DMON)] == 2
    assert alert_lvls[int(10 / DT_DMON)] == 3
    assert isinstance(d_status.awareness, float)

  # engaged, distracted past red and beyond the no-response window -> unavailability response + lockout
  def test_distracted_lockout(self):
    alert_lvls, d_status = self._run_seq(always_distracted, always_false, always_true, always_false)
    assert alert_lvls[int(DISTRACTED_SECONDS_TO_RED / DT_DMON)] == 3
    assert d_status.lockout_active
    assert d_status.lockout_time_elapsed > 0
    assert d_status.lockout_count >= 1

  # no face -> wheeltouch red, sustained past the no-response timeout -> unavailability response + lockout
  def test_invisible_lockout(self):
    _, d_status = self._run_seq(always_no_face, always_false, always_true, always_false)
    assert d_status.active_policy == log.DriverMonitoringState.MonitoringPolicy.wheeltouch
    assert d_status.lockout_active
    assert d_status.lockout_count >= 1

  # engaged, no face detected the whole time, no action
  def test_fully_invisible_driver(self):
    alert_lvls, d_status = self._run_seq(always_no_face, always_false, always_true, always_false)
    s = d_status.settings
    assert alert_lvls[int(s._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT / 2 / DT_DMON)] == 0
    assert alert_lvls[int((s._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT + \
                    (s._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT - s._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT) / 2) / DT_DMON)] == 1
    assert alert_lvls[int((s._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT + \
                    (s._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT - s._WHEELTOUCH_POLICY_ALERT_2_TIMEOUT) / 2) / DT_DMON)] == 2
    assert alert_lvls[int((s._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT + \
                    (TEST_TIMESPAN - 10 - s._WHEELTOUCH_POLICY_ALERT_3_TIMEOUT) / 2) / DT_DMON)] == 3
    assert d_status.active_policy == log.DriverMonitoringState.MonitoringPolicy.wheeltouch

  # engaged, down to orange via the drowsy fast path, then attentive
  #  - drowsiness latch briefly holds the escalation, then awareness recovers to green
  def test_normal_driver(self):
    ds_vector = [msg_DISTRACTED] * int(6/DT_DMON) + \
                [msg_ATTENTIVE] * int((TEST_TIMESPAN-6)/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, always_false, always_true, always_false)
    assert alert_lvls[int(3/DT_DMON)] == 0
    assert alert_lvls[int(5.9/DT_DMON)] == 2
    assert alert_lvls[int(10/DT_DMON)] == 2
    assert alert_lvls[int(40/DT_DMON)] == 0

  # engaged, down to orange, driver dodges camera, then comes back still distracted, down to red, \
  #                          driver dodges, and then touches wheel to no avail, disengages and reengages
  #  - orange/red alert should remain after disappearance, and only disengaging clears red
  def test_biggest_comma_fan(self):
    _orange_time, _red_time = 6., 8.5  # drowsy fast-path ladder (stock: 9s/14s)
    _invisible_time = 2  # seconds
    ds_vector = always_distracted[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(_orange_time/DT_DMON):int((_orange_time+_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    ds_vector[int((_red_time+_invisible_time)/DT_DMON):int((_red_time+2*_invisible_time)/DT_DMON)] \
                                                        = [msg_NO_FACE_DETECTED] * int(_invisible_time/DT_DMON)
    interaction_vector[int((_red_time+2*_invisible_time+0.5)/DT_DMON):int((_red_time+2*_invisible_time+1.5)/DT_DMON)] \
                                                        = [True] * int(1/DT_DMON)
    op_vector[int((_red_time+2*_invisible_time+2.5)/DT_DMON):int((_red_time+2*_invisible_time+3)/DT_DMON)] \
                                                        = [False] * int(0.5/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert alert_lvls[int((_orange_time+0.5*_invisible_time)/DT_DMON)] == 2
    assert alert_lvls[int((_red_time+1.5*_invisible_time)/DT_DMON)] == 3
    assert alert_lvls[int((_red_time+2*_invisible_time+1.5)/DT_DMON)] == 3
    assert alert_lvls[int((_red_time+2*_invisible_time+3.5)/DT_DMON)] == 0

  # engaged, invisible driver, down to orange, driver touches wheel; then down to orange again, driver appears
  #  - both actions should clear the alert, but momentary appearance should not
  def test_sometimes_transparent_commuter(self):
    for _visible_time in (0.5, 10):
      ds_vector = always_no_face[:]*2
      interaction_vector = always_false[:]*2
      ds_vector[int((2*INVISIBLE_SECONDS_TO_ORANGE+1)/DT_DMON):int((2*INVISIBLE_SECONDS_TO_ORANGE+1+_visible_time)/DT_DMON)] = \
                                                                                               [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
      interaction_vector[int((INVISIBLE_SECONDS_TO_ORANGE)/DT_DMON):int((INVISIBLE_SECONDS_TO_ORANGE+1)/DT_DMON)] = [True] * int(1/DT_DMON)
      alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, 2*always_true, 2*always_false)
      assert alert_lvls[int(dm_settings._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT/2/DT_DMON)] == 0
      assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE-0.1)/DT_DMON)] == 2
      assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE+0.1)/DT_DMON)] == 0
      if _visible_time == 0.5:
        assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE*2+1-0.1)/DT_DMON)] == 2
        assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE*2+1+0.1+_visible_time)/DT_DMON)] == 2
      elif _visible_time == 10:
        assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE*2+1-0.1)/DT_DMON)] == 2
        assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE*2+1+0.1+_visible_time)/DT_DMON)] == 0

  # engaged, invisible driver, down to red, driver appears and then touches wheel, then disengages/reengages
  #  - only disengage will clear the alert
  def test_last_second_responder(self):
    _visible_time = 2  # seconds
    ds_vector = always_no_face[:]
    interaction_vector = always_false[:]
    op_vector = always_true[:]
    ds_vector[int(INVISIBLE_SECONDS_TO_RED/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time)/DT_DMON)] = [msg_ATTENTIVE] * int(_visible_time/DT_DMON)
    interaction_vector[int((INVISIBLE_SECONDS_TO_RED+_visible_time)/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time+1)/DT_DMON)] = [True] * int(1/DT_DMON)
    op_vector[int((INVISIBLE_SECONDS_TO_RED+_visible_time+1)/DT_DMON):int((INVISIBLE_SECONDS_TO_RED+_visible_time+0.5)/DT_DMON)] = [False] * int(0.5/DT_DMON)
    alert_lvls, _ = self._run_seq(ds_vector, interaction_vector, op_vector, always_false)
    assert alert_lvls[int(dm_settings._WHEELTOUCH_POLICY_ALERT_1_TIMEOUT/2/DT_DMON)] == 0
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE-0.1)/DT_DMON)] == 2
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_RED-0.1)/DT_DMON)] == 3
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_RED+0.5*_visible_time)/DT_DMON)] == 3
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_RED+_visible_time+0.5)/DT_DMON)] == 3
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_RED+_visible_time+1+0.1)/DT_DMON)] == 0

  # disengaged, always distracted driver
  #  - dm should stay quiet when not engaged
  def test_pure_dashcam_user(self):
    alert_lvls, _ = self._run_seq(always_distracted, always_false, always_false, always_false)
    assert all(a == 0 for a in alert_lvls)

  # engaged, car stops at traffic light, down to orange, no action, then car starts moving
  #  - should only reach green when stopped, but continues counting down on launch
  def test_long_traffic_light_victim(self):
    _redlight_time = 60  # seconds
    lowspeed_vector = always_true[:]
    lowspeed_vector[int(_redlight_time/DT_DMON):] = [False] * int((TEST_TIMESPAN-_redlight_time)/DT_DMON)
    alert_lvls, d_status = self._run_seq(always_distracted, always_false, always_true, lowspeed_vector)
    s = d_status.settings
    # stopped: green at most (the drowsy fast-path hovers at the green threshold), never orange/red
    assert alert_lvls[int((_redlight_time-0.1)/DT_DMON)] <= 1
    _alert_1_to_2 = s._VISION_POLICY_ALERT_2_TIMEOUT - s._VISION_POLICY_ALERT_1_TIMEOUT
    assert alert_lvls[int((_redlight_time+0.5)/DT_DMON)] == 1
    assert alert_lvls[int((_redlight_time+_alert_1_to_2+0.5)/DT_DMON)] == 2

  # engaged, distracted while moving, then car stops after reaching orange
  #  - should reset timer to pre green at low speed
  def test_distracted_then_stops(self):
    _stop_time = 6  # stop while orange (drowsy fast-path reds at ~8.5s)
    lowspeed_vector = always_false[:]
    lowspeed_vector[int(_stop_time/DT_DMON):] = [True] * int((TEST_TIMESPAN-_stop_time)/DT_DMON)
    alert_lvls, _ = self._run_seq(always_distracted, always_false, always_true, lowspeed_vector)
    # just before and briefly after stopping: orange alert; goes away quickly after stopped
    assert alert_lvls[int((_stop_time+0.1)/DT_DMON)] == 2
    assert alert_lvls[int((_stop_time+0.5)/DT_DMON)] == 0

  # engaged, model is somehow uncertain and driver is distracted
  #  - should fall back to wheel touch after uncertain alert
  def test_somehow_indecisive_model(self):
    ds_vector = [msg_DISTRACTED_BUT_SOMEHOW_UNCERTAIN] * int(TEST_TIMESPAN/DT_DMON)
    interaction_vector = always_false[:]
    alert_lvls, d_status = self._run_seq(ds_vector, interaction_vector, always_true, always_false)
    s = d_status.settings
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE-1+DT_DMON*s._HI_STD_FALLBACK_TIME-0.1)/DT_DMON)] == 1
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_ORANGE-1+DT_DMON*s._HI_STD_FALLBACK_TIME+0.1)/DT_DMON)] == 2
    assert alert_lvls[int((INVISIBLE_SECONDS_TO_RED-1+DT_DMON*s._HI_STD_FALLBACK_TIME+0.1)/DT_DMON)] == 3

  # engaged, driver has heavy eyelids (partial closure below the stock blink threshold)
  #  - previously invisible to the monitor, now flagged drowsy and escalated ~2x faster
  def test_drowsy_heavy_eyelids(self):
    eyelids = [make_msg_eyelids(0.6)] * int(TEST_TIMESPAN / DT_DMON)
    alert_lvls, d_status = self._run_seq(eyelids, always_false, always_true, always_false)
    assert not d_status.distracted_types['eye']  # stock eye detector does not fire at 0.6
    assert d_status.drowsy_latched
    assert alert_lvls[int(11 / DT_DMON)] == 3  # red at ~9.5s instead of the stock ~13s
    assert d_status.lockout_active

  # engaged, driver nods off repeatedly (the classic falling-asleep head nod)
  #  - 3+ nods in the window flag drowsiness and escalate
  def test_drowsy_nodding(self):
    down = make_msg_pitch(-0.25)
    up = make_msg_pitch(0.)
    pattern = [down] * int(0.5 / DT_DMON) + [up] * int(2.5 / DT_DMON)
    nods = (pattern * (int(TEST_TIMESPAN / 3) + 1))[:int(TEST_TIMESPAN / DT_DMON)]
    alert_lvls, d_status = self._run_seq(nods, always_false, always_true, always_false)
    assert len(d_status.nod_steps) >= 3
    assert not d_status.distracted_types['pose']  # nods stay below the stock pose threshold
    assert d_status.drowsy_latched
    assert alert_lvls[int(25 / DT_DMON)] == 3  # red at ~17s

  # engaged, driver's head has fallen forward and stays there
  def test_drowsy_head_droop(self):
    droop = [make_msg_pitch(-0.35)] * int(TEST_TIMESPAN / DT_DMON)
    alert_lvls, d_status = self._run_seq(droop, always_false, always_true, always_false)
    assert not d_status.distracted_types['pose']  # below the stock pose threshold
    assert d_status.drowsy_latched
    assert alert_lvls[int(15 / DT_DMON)] == 3  # red at ~12s

  # engaged, repeated 2s microsleeps (eyes 85% closed) every 10s
  #  - each burst is too short for the stock detectors, but PERCLOS catches the pattern
  def test_drowsy_perclos_bursts(self):
    closed = make_msg_eyelids(0.85)
    open_ = make_msg_eyelids(0.)
    pattern = [closed] * int(2 / DT_DMON) + [open_] * int(8 / DT_DMON)
    bursts = (pattern * (int(TEST_TIMESPAN / 10) + 1))[:int(TEST_TIMESPAN / DT_DMON)]
    alert_lvls, d_status = self._run_seq(bursts, always_false, always_true, always_false)
    assert not d_status.distracted_types['eye']  # 0.85 never trips the stock eye detector
    assert d_status.drowsy_latched
    assert alert_lvls[int(45 / DT_DMON)] == 3  # PERCLOS trips at ~30s, red at ~38s
