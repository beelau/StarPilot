import pyray as rl
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.selfdrive.ui.onroad.starpilot.widgets.base import LayoutWidget
from openpilot.selfdrive.ui.onroad.starpilot.delivery_button import DeliveryButton

class DeliveryButtonWidget(LayoutWidget):
  def __init__(self):
    super().__init__("delivery_button", priority=1)
    self._button = DeliveryButton()
    self._button.set_visible(True)
    self._child(self._button)

  @property
  def is_visible(self) -> bool:
    return bool(ui_state.started)

  def get_size(self) -> tuple[float, float]:
    return 192.0, 192.0

  def _render(self, rect: rl.Rectangle) -> None:
    # Render the child button within the layout rect
    self._button.render(rect)
