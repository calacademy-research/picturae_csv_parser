import tkinter as tk
import importlib


class CsvCreateBase(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Create Import CSV")
        self.geometry("800x600")

        self.container = tk.Frame(self)
        self.container.pack(side="top", fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        self.available_frames = {
            "CsvCreatePIC": ("pic_csv_prompt", "CsvCreatePIC"),
            "CSVEditorApp": ("csv_editor", "CSVEditorApp")
        }
        self.show_frame("CsvCreatePIC")

    def show_frame(self, frame_name, csv_path=None, callback=False):
        # Hide all existing frames
        for frame in self.frames.values():
            frame.pack_forget()

        module_name, class_name = self.available_frames[frame_name]
        frame_module = importlib.import_module(module_name)
        frame_class = getattr(frame_module, class_name)

        # Check if the frame needs to be initialized or updated
        if frame_name not in self.frames:  # Initialize the frame if it's not already done
            if frame_name == "CSVEditorApp":
                frame = frame_class(csv_path, self.container, self)
            else:
                frame = frame_class(self.container, self, callback)
            self.frames[frame_name] = frame
        else:
            frame = self.frames[frame_name]
            if csv_path and hasattr(frame, 'update_with_csv_path'):  # Update the frame with new csv_path if provided
                frame.update_with_csv_path(csv_path)

        # Show the requested frame
        frame.pack(fill="both", expand=True)

if __name__ == "__main__":
    app = CsvCreateBase()
    app.mainloop()
