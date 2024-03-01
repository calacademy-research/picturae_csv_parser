import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import csv

class CSVEditorApp(tk.Frame):
    def __init__(self, csv_path, parent, controller, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.csv_path = csv_path
        self.parent = parent
        self.controller = controller
        # Removed the title and geometry settings as those apply to top-level windows
        self.edit_history = []
        self.tree = None
        self.load_csv()

    def load_csv(self):
        # Frame for the search bar
        self.search_frame = tk.Frame(self)
        self.search_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5, anchor='nw')

        # Label for the search bar
        self.search_label = tk.Label(self.search_frame, text="Search:")
        self.search_label.pack(side=tk.LEFT, padx=5)

        # Search Entry
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(self.search_frame, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT)
        self.search_entry.bind('<Return>', lambda event: self.search_rows())

        with open(self.csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)
            headers = data[0]
            content = data[1:]

        self.tree = ttk.Treeview(self, columns=headers, show="headings")
        self.tree.pack(expand=True, fill=tk.BOTH)

        for header in headers:
            self.tree.heading(header, text=header)
            self.tree.column(header, width=100)

        for row in content:
            self.tree.insert('', tk.END, values=row)

        # Add vertical scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Add horizontal scrollbar
        scrollbar = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(xscroll=scrollbar.set)
        scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Frame for Save Button
        self.save_button_frame = tk.Frame(self)
        self.save_button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        # Inner frame to hold buttons and center them
        self.inner_button_frame = tk.Frame(self.save_button_frame)
        self.inner_button_frame.pack(expand=True, pady=10)

        # Save Button within the inner frame
        self.save_button = ttk.Button(self.inner_button_frame, text="Save", command=self.save_csv)
        self.save_button.pack(side=tk.LEFT, padx=10, pady=10)

        # New "Save and Continue" button
        self.save_continue_button = ttk.Button(self.inner_button_frame, text="Save and Continue",
                                               command=self.save_and_continue)
        self.save_continue_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Exit Button within the inner frame
        self.exit_button = ttk.Button(self.inner_button_frame, text="Exit", command=self.exit_program)
        self.exit_button.pack(side=tk.LEFT, padx=10, pady=10)

        # undo ctrl z
        self.bind_all('<Control-z>', lambda event: self.undo_last_edit())

        # Custom scroll event bindings
        self.tree.bind("<MouseWheel>", self.on_vertical_scroll)
        self.tree.bind("<Shift-MouseWheel>", self.on_horizontal_scroll)

        # Bind double click to edit cell
        self.tree.bind('<Double-1>', self.edit_cell)

    def search_rows(self):
        query = self.search_var.get().lower()
        for child in self.tree.get_children():
            # Assuming the search is for any cell value in the row
            if query in " ".join(self.tree.item(child, 'values')).lower():
                self.tree.selection_set(child)
                self.tree.see(child)
                return
        messagebox.showinfo("Search", "No matches found.")

    def edit_cell(self, event):
        row_id = self.tree.identify_row(event.y)
        column = self.tree.identify_column(event.x)

        if not row_id or not column:
            return

        # Clear any existing entry widget
        for widget in self.tree.place_slaves():
            widget.destroy()

        # Get the column index
        col_index = self.tree.heading(column, 'text')

        # Determine the position of the cell
        x, y, width, height = self.tree.bbox(row_id, column)

        # Create an Entry widget
        entry_edit = tk.Entry(self.tree, width=width)
        entry_edit.place(x=x, y=y, width=width, height=height)

        # Set the entry widget's content to the cell's current value
        cell_value = self.tree.item(row_id, 'values')[int(column[1:]) - 1]
        entry_edit.insert(0, cell_value)

        # Function to save the edited value

        def save_edit(event=None):
            # Retrieve the new value from the Entry widget
            new_value = entry_edit.get()
            # Check if the new value is different from the old value to avoid unnecessary history entries
            if new_value != cell_value:
                # Update the cell value and capture the edit for undo functionality
                self.update_cell_value(row_id, column, new_value)
            entry_edit.destroy()

        def cancel_edit(event=None):
            entry_edit.destroy()

        entry_edit.bind('<Escape>', cancel_edit)
        entry_edit.bind('<FocusOut>', save_edit)

        entry_edit.focus_set()

    def update_cell_value(self, row_id, column, new_value):
        old_value = self.tree.set(row_id, column)
        if old_value != new_value:
            self.edit_history.append((row_id, column, old_value, new_value))
            self.tree.set(row_id, column, new_value)

    def undo_last_edit(self):
        if self.edit_history:
            # retrieve last edit from history
            row_id, column, old_value, _ = self.edit_history.pop()
            # Revert the cell to its old value
            self.tree.set(row_id, column, old_value)
        else:
            messagebox.showinfo("Undo", "No more actions to undo.")

    def on_vertical_scroll(self, event):
        if event.delta > 0:
            self.tree.yview_scroll(-6, "units")  # Move up 3 units
        else:
            self.tree.yview_scroll(6, "units")  # Move down 3 units

    def on_horizontal_scroll(self, event):
        if event.delta > 0:
            self.tree.xview_scroll(-6, "units")  # Move left 3 units
        else:
            self.tree.xview_scroll(6, "units")  # Move right 3 units

    def save_csv(self):
        # Open the file in write mode
        with open(self.csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write the headers
            headers = [self.tree.heading(col)['text'] for col in self.tree['columns']]
            writer.writerow(headers)

            # Write the content
            for child in self.tree.get_children():
                row_values = [self.tree.set(child, col) for col in self.tree['columns']]
                writer.writerow(row_values)

        messagebox.showinfo("Save", "Changes saved successfully!")

    def save_and_continue(self):
        # Call the existing save logic
        self.save_csv()
        # Close the application window
        self.controller.show_frame("CsvCreatePIC", callback=True)

    def exit_program(self):
        if messagebox.askyesno("Exit", "Are you sure you want to exit?"):
            self.controller.destroy()



        # Example usage
# if __name__ == "__main__":
#     app = CSVEditorApp("picturae_csv/20230628/picturae_folder(20230628).csv")
#     app.mainloop()
