import pyray as rl
from openpilot.common.params import Params
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

BTN_SIZE = 192


class DeliveryButton(Widget):
  """One-tap Delivery Mode toggle on the driving screen.

  No need to dig through Galaxy settings: tap to switch the lenient
  phone-detection profile on/off. Green = delivery mode active.
  """

  def __init__(self):
    super().__init__()
    self._params = Params()
    self._font_bold = gui_app.font(FontWeight.BOLD)
    self._rect = rl.Rectangle(0, 0, BTN_SIZE, BTN_SIZE)
    self._delivery_mode = False

    self.set_visible(lambda: ui_state.started)

  def _update_state(self):
    self._delivery_mode = self._params.get_bool("DeliveryMode")

  def _handle_mouse_release(self, _):
    self._params.put_bool("DeliveryMode", not self._delivery_mode)
    self._delivery_mode = not self._delivery_mode

  def _render(self, rect: rl.Rectangle):
    self._rect = rect
    cx = int(rect.x + rect.width / 2)
    cy = int(rect.y + rect.height / 2)

    if self._delivery_mode:
      bg = rl.Color(0, 180, 90, 220)
      fg = rl.Color(255, 255, 255, 255)
      state_text, state_size = "ON", 34
    else:
      bg = rl.Color(20, 20, 20, 140)
      fg = rl.Color(160, 160, 160, 255)
      state_text, state_size = "OFF", 34

    if self.is_pressed:
      bg.a = max(0, bg.a - 60)

    rl.draw_circle(cx, cy, rect.width / 2, bg)

    label, label_size = "DELIVERY", 30
    label_w = measure_text_cached(self._font_bold, label, label_size).x
    rl.draw_text_ex(self._font_bold, label, rl.Vector2(cx - label_w / 2, cy - 34), label_size, 0, fg)
    state_w = measure_text_cached(self._font_bold, state_text, state_size).x
    rl.draw_text_ex(self._font_bold, state_text, rl.Vector2(cx - state_w / 2, cy + 4), state_size, 0, fg)
