import tkinter as tk
from tkinter import messagebox
import subprocess

def run_csv_create():
    date = date_entry.get()

    try:
        subprocess.run(['python', 'picturae_csv_create.py', '-d', date])
        messagebox.showinfo("Success", "CSV created successfully in batch folder")
    except ValueError as e:
        messagebox.showerror("Error", str(e))

app = tk.Tk()

app.title("Create Picturae CSV")

date_label = tk.Label(app, text="Enter date (YYYYMMDD)")

date_label.pack()

date_entry = tk.Entry(app)
date_entry.pack()

run_button = tk.Button(app, text="Create CSV", command=run_csv_create)

run_button.pack()

app.mainloop()
