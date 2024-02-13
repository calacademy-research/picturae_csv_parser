import tkinter as tk
from tkinter import messagebox
import subprocess
import csv
from csv_editor import CSVEditorApp

def run_csv_create():
    date = date_entry.get()
    process = subprocess.Popen(['python', 'picturae_csv_create.py', '-d', date], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        error_message = stderr.decode("utf-8").strip()
        line = error_message.splitlines()[-1]
        try:
            if "ValueError" in line:
                if line == "ValueError: Taxonomic name with 2 missing ranks":
                    messagebox.showerror("Error:", line)
                    CSVEditorApp(csv_path=f"picturae_csv/{date}/picturae_folder({date}).csv")
                else:
                    messagebox.showerror("Error:", line)
            else:
                messagebox.showerror("Error:", line)
        except Exception as e:
            messagebox.showerror("Error Opening CSV", str(e))
    else:

        messagebox.showinfo("Success", "CSV created successfully in batch folder")


app = tk.Tk()
app.title("Create Picturae CSV")

instruction_text = "Please enter the date as found in the batch folder name: CP1_YYYYMMDD_BATCH_0001."
instruction_label = tk.Label(app, text=instruction_text, wraplength=780)  # Wrap text to fit window
instruction_label.pack(pady=10)  # Add some padding for aesthetics

app.geometry("800x600")

date_label = tk.Label(app, text="Enter date (YYYYMMDD)")
date_label.pack()

date_entry = tk.Entry(app)
date_entry.pack()

run_button = tk.Button(app, text="Create CSV", command=run_csv_create)
run_button.pack()

app.mainloop()
