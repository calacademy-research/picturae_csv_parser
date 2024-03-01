import tkinter as tk
from tkinter import messagebox
import subprocess
# Assuming master_run() and CSVEditorApp are defined in the commented-out imports
# from specify7_ipup import master_run
# from csv_editor import CSVEditorApp

class CsvCreatePIC(tk.Frame):
    def __init__(self, parent, controller, callback=False, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.controller = controller
        self.callback = callback
        self.parent = parent

        self.setup_ui()


    def setup_ui(self):
        instruction_text = "Please enter the date as found in the batch folder name: CP1_YYYYMMDD_BATCH_0001."
        instruction_label = tk.Label(self.master, text=instruction_text, wraplength=780)
        instruction_label.pack(pady=10)

        date_label = tk.Label(self.master, text="Enter date (YYYYMMDD)")
        date_label.pack()

        date_entry = tk.Entry(self.master)
        date_entry.pack()

        run_button = tk.Button(self.master, text="Create CSV", command=lambda: self.run_csv_create(date_entry))
        run_button.pack()

        if self.callback is True:
            date_entry = date_entry.get()
            self.run_csv_create(date_entry=date_entry)


    def run_csv_create(self, date_entry):
        if not isinstance(date_entry, str):
            self.date = date_entry.get()
        else:
            self.date = date_entry
        process = subprocess.Popen(['python', 'picturae_csv_create.py', '-d', self.date], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            error_message = stderr.decode("utf-8").strip()
            line = error_message.splitlines()[-1]
            try:
                if "ValueError" in line:
                    if "missing ranks" in line:
                        messagebox.showerror("Error:", line)
                        csv_path = f"picturae_csv/{self.date}/picturae_folder({self.date}).csv"
                        self.open_csv_editor(path=csv_path)
                    elif "missing country" in line:
                        pass
                    elif "missing column" in line:
                        pass
                    elif "un-annotated duplicate barcode" in line:
                        pass
                    else:
                        messagebox.showerror("Error:", line)
                else:
                    messagebox.showerror("Error:", line)
            except Exception as e:
                messagebox.showerror("Error Opening CSV", str(e))
        else:
            messagebox.showinfo("Success", "CSV created successfully in batch folder")

    def run_main_as_callback(self):
        # self.master.title("Create Picturae CSV")
        self.unhide_main_app_widgets()
        self.setup_ui(date_override=self.date)
        self.run_csv_create(date_entry=self.date)

    def hide_main_app_widgets(self):
        self.original_pack_config = []  # List to store original pack configurations
        for widget in self.parent.winfo_children():
            # Get current pack_info for the widget
            config = widget.pack_info()
            self.original_pack_config.append((widget, config))
            widget.pack_forget()

    def unhide_main_app_widgets(self):
        for widget, config in self.original_pack_config:
            widget.pack(**config)  # Unpack the config dict to use as pack parameters
        self.original_pack_config.clear()


    def open_csv_editor(self, path):
       self.controller.show_frame("CSVEditorApp", path)

    def auto_run_script(self):
        # Directly call the button's command
        self.run_csv_create(date_entry=self.date)


# if __name__ == "__main__":
#     root = tk.Tk()
#     root.title("Create Picturae CSV")
#     csv_create = CsvCreatePIC(master=root)
#     root.mainloop()
