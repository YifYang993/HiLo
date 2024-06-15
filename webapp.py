from apps.infer_interface import HiLoInfer
from ui_args import ui_infer_args
import gradio as gio
import numpy as np
from types import SimpleNamespace


class HiloWebApp:
    def __init__(self):
        self.infer_interface = HiLoInfer(args=SimpleNamespace(**ui_infer_args),
                                         on_infer_status_update=self._update_status)
        self.ui = gio.Interface(fn=self._infer_worker,
                                inputs="image",
                                outputs=[
                                    gio.Image(label="smpl"),
                                    gio.Image(label="overlap"),
                                    gio.Model3D(label="refine_model",
                                                clear_color=(
                                                    0.0, 0.0, 0.0, 0.0)),
                                    gio.Model3D(label="smpl_model",
                                                clear_color=(
                                                    0.0, 0.0, 0.0, 0.0))
                                ],
                                allow_flagging='never').queue()

    def _infer_worker(self, input_image: np.ndarray):
        dataset = self.infer_interface.prepare_dataset([input_image])
        result = self.infer_interface.infer(dataset[0], dataset)

        return result['smpl'], result['overlap'], result['refine_obj_file'], result['smpl_obj_file']

    def _update_status(self, desc, prog):
        pass

    def launch(self):
        self.ui.launch()
        pass


if __name__ == '__main__':
    app = HiloWebApp()
    app.launch()
